use proc_macro2::TokenStream;
use quote::quote;
use std::{collections::HashMap, iter};
use syn::{
    parse::Parser,
    punctuated::Punctuated,
    token::Comma,
    visit_mut::{self, VisitMut},
    Expr, ExprCall, ExprParen, ExprPath, ExprReference, Ident, Macro, Path, PathSegment,
};

struct MigrationCtx(HashMap<&'static str, &'static str>);

impl visit_mut::VisitMut for MigrationCtx {
    fn visit_macro_mut(&mut self, i: &mut syn::Macro) {
        if let Ok(mut expr) = i.parse_body_with(Punctuated::<Expr, Comma>::parse_terminated) {
            for expr in expr.iter_mut() {
                self.visit_expr_mut(expr);
            }

            *i = Macro {
                path: i.path.clone(),
                bang_token: i.bang_token,
                delimiter: i.delimiter.clone(),
                tokens: quote! { #expr },
            };
        }
    }

    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        visit_mut::visit_expr_mut(self, i);

        match i {
            Expr::MethodCall(call) if call.method == "faer_add" => {
                *i = Expr::Binary(syn::ExprBinary {
                    attrs: vec![],
                    left: call.receiver.clone(),
                    op: syn::BinOp::Add(Default::default()),
                    right: Box::new(call.args[0].clone()),
                });
                *i = Expr::Paren(ExprParen {
                    attrs: vec![],
                    paren_token: Default::default(),
                    expr: Box::new(i.clone()),
                });
            }
            Expr::MethodCall(call) if call.method == "faer_sub" => {
                *i = Expr::Binary(syn::ExprBinary {
                    attrs: vec![],
                    left: call.receiver.clone(),
                    op: syn::BinOp::Sub(Default::default()),
                    right: Box::new(call.args[0].clone()),
                });
                *i = Expr::Paren(ExprParen {
                    attrs: vec![],
                    paren_token: Default::default(),
                    expr: Box::new(i.clone()),
                });
            }
            Expr::MethodCall(call) if call.method == "faer_mul" => {
                *i = Expr::Binary(syn::ExprBinary {
                    attrs: vec![],
                    left: call.receiver.clone(),
                    op: syn::BinOp::Mul(Default::default()),
                    right: Box::new(call.args[0].clone()),
                });
                *i = Expr::Paren(ExprParen {
                    attrs: vec![],
                    paren_token: Default::default(),
                    expr: Box::new(i.clone()),
                });
            }
            Expr::MethodCall(call) if call.method == "faer_div" => {
                *i = Expr::Binary(syn::ExprBinary {
                    attrs: vec![],
                    left: call.receiver.clone(),
                    op: syn::BinOp::Div(Default::default()),
                    right: Box::new(call.args[0].clone()),
                });
                *i = Expr::Paren(ExprParen {
                    attrs: vec![],
                    paren_token: Default::default(),
                    expr: Box::new(i.clone()),
                });
            }
            Expr::MethodCall(call) if call.method == "faer_neg" => {
                *i = Expr::Unary(syn::ExprUnary {
                    attrs: vec![],
                    op: syn::UnOp::Neg(Default::default()),
                    expr: call.receiver.clone(),
                });
                *i = Expr::Paren(ExprParen {
                    attrs: vec![],
                    paren_token: Default::default(),
                    expr: Box::new(i.clone()),
                });
            }

            Expr::MethodCall(call) if call.method.to_string().starts_with("faer_") => {
                if let Some(new_method) = self.0.get(&*call.method.to_string()).map(|x| &**x) {
                    *i = math_expr(
                        &Ident::new(new_method, call.method.span()),
                        std::iter::once(&*call.receiver).chain(call.args.iter()),
                    )
                }
            }

            _ => {}
        }
    }
}

struct MathCtx;

fn ident_expr(ident: &syn::Ident) -> Expr {
    Expr::Path(ExprPath {
        attrs: vec![],
        qself: None,
        path: Path {
            leading_colon: None,
            segments: Punctuated::from_iter(iter::once(PathSegment {
                ident: ident.clone(),
                arguments: syn::PathArguments::None,
            })),
        },
    })
}

impl visit_mut::VisitMut for MathCtx {
    fn visit_macro_mut(&mut self, i: &mut syn::Macro) {
        if let Ok(mut expr) = i.parse_body_with(Punctuated::<Expr, Comma>::parse_terminated) {
            for expr in expr.iter_mut() {
                self.visit_expr_mut(expr);
            }

            *i = Macro {
                path: i.path.clone(),
                bang_token: i.bang_token,
                delimiter: i.delimiter.clone(),
                tokens: quote! { #expr },
            };
        }
    }

    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        visit_mut::visit_expr_mut(self, i);

        match i {
            Expr::Unary(unary) => match unary.op {
                syn::UnOp::Neg(minus) => {
                    *i = Expr::Call(ExprCall {
                        attrs: vec![],
                        func: Box::new(ident_expr(&Ident::new("neg", minus.span))),
                        paren_token: Default::default(),
                        args: std::iter::once((*unary.expr).clone())
                            .map(|e| {
                                Expr::Reference(ExprReference {
                                    attrs: vec![],
                                    and_token: Default::default(),
                                    mutability: None,
                                    expr: Box::new(e),
                                })
                            })
                            .collect(),
                    })
                }
                _ => {}
            },
            Expr::Binary(binop) => {
                let func = match binop.op {
                    syn::BinOp::Add(plus) => Some(Ident::new("add", plus.span)),
                    syn::BinOp::Sub(minus) => Some(Ident::new("sub", minus.span)),
                    syn::BinOp::Mul(star) => Some(Ident::new("mul", star.span)),
                    syn::BinOp::Div(star) => Some(Ident::new("div", star.span)),
                    _ => None,
                };
                if let Some(func) = func {
                    *i = Expr::Call(ExprCall {
                        attrs: vec![],
                        func: Box::new(ident_expr(&func)),
                        paren_token: Default::default(),
                        args: [(*binop.left).clone(), (*binop.right).clone()]
                            .into_iter()
                            .map(|e| {
                                Expr::Reference(ExprReference {
                                    attrs: vec![],
                                    and_token: Default::default(),
                                    mutability: None,
                                    expr: Box::new(e),
                                })
                            })
                            .collect(),
                    })
                }
            }

            Expr::Call(call) => match &*call.func {
                Expr::Path(e) if e.path.get_ident().is_some() => {
                    let name = &*e.path.get_ident().unwrap().to_string();
                    if matches!(
                        name,
                        "sqrt"
                            | "from_real"
                            | "copy"
                            | "max"
                            | "min"
                            | "conj"
                            | "absmax"
                            | "abs2"
                            | "abs1"
                            | "abs"
                            | "add"
                            | "sub"
                            | "div"
                            | "mul"
                            | "mul_real"
                            | "mul_pow2"
                            | "hypot"
                            | "neg"
                            | "recip"
                            | "real"
                            | "imag"
                            | "is_nan"
                            | "is_finite"
                            | "is_zero"
                            | "lt_zero"
                            | "gt_zero"
                            | "le_zero"
                            | "ge_zero"
                    ) {
                        call.args.iter_mut().for_each(|x| {
                            *x = Expr::Reference(ExprReference {
                                attrs: vec![],
                                and_token: Default::default(),
                                mutability: None,
                                expr: Box::new(x.clone()),
                            })
                        })
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
}

fn math_expr<'a>(method: &Ident, args: impl Iterator<Item = &'a Expr>) -> Expr {
    Expr::Call(ExprCall {
        attrs: vec![],

        paren_token: Default::default(),
        args: args.cloned().collect(),
        func: Box::new(ident_expr(method)),
    })
}

#[proc_macro_attribute]
pub fn math(_: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let Ok(mut item) = syn::parse::<syn::ItemFn>(item.clone()) else {
        return item;
    };
    let mut rust_ctx = MathCtx;
    rust_ctx.visit_item_fn_mut(&mut item);
    let item = quote! { #item };
    item.into()
}

#[proc_macro_attribute]
pub fn migrate(
    _: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let Ok(mut item) = syn::parse::<syn::ItemFn>(item.clone()) else {
        return item;
    };

    let mut rust_ctx = MigrationCtx(
        [
            //
            ("faer_add", "add"),
            ("faer_sub", "sub"),
            ("faer_mul", "mul"),
            ("faer_div", "div"),
            ("faer_neg", "neg"),
            ("faer_inv", "recip"),
            ("faer_abs", "abs"),
            ("faer_abs2", "abs2"),
            ("faer_sqrt", "sqrt"),
            ("faer_conj", "conj"),
            ("faer_real", "real"),
            ("faer_scale_real", "mul_real"),
            ("faer_scale_power_of_two", "mul_pow2"),
        ]
        .into_iter()
        .collect(),
    );
    rust_ctx.visit_item_fn_mut(&mut item);
    let mut rust_ctx = MathCtx;
    rust_ctx.visit_item_fn_mut(&mut item);

    let item = quote! { #item };
    item.into()
}

struct Tree {
    name: Ident,
    children: Vec<Tree>,
}

impl Tree {
    fn parse_expr(e: &Expr) -> Self {
        match e {
            Expr::Call(call) if matches!(&*call.func, Expr::Path(path) if path.path.get_ident().is_some()) =>
            {
                let name = match &*call.func {
                    Expr::Path(path) => path.path.get_ident().unwrap().clone(),
                    _ => panic!(),
                };
                let children = call.args.iter().map(|arg| Self::parse_expr(arg)).collect();

                Self { name, children }
            }
            Expr::Path(path) if path.path.get_ident().is_some() => Self {
                name: path.path.get_ident().unwrap().clone(),
                children: vec![],
            },
            _ => {
                panic!();
            }
        }
    }

    fn init(&self) -> TokenStream {
        let name = &self.name;
        let glue_name = Ident::new(&(format!("_{name}_")), name.span());
        let mut stmt = quote! { let mut #glue_name = crate::hacks::NonCopy; let #name = crate::hacks::__with_lifetime_of(&mut #glue_name); };

        for child in &self.children {
            let child = child.init();
            stmt = quote! { #stmt #child };
        }

        stmt
    }

    fn deinit(&self) -> TokenStream {
        let name = &self.name;
        let glue_name = Ident::new(&(format!("_{name}_")), name.span());

        let mut stmt = quote! {};
        for child in &self.children {
            let child = child.deinit();
            stmt = quote! { #stmt #child };
        }
        quote! {  #stmt ::core::mem::drop(#name); ::core::mem::drop(&#glue_name);}
    }

    fn struct_def(&self) -> TokenStream {
        let name = &self.name;
        let children = self.children.iter().map(|x| &x.name);
        let ty = children.clone();

        let mut stmt = quote! {
            #[allow(non_camel_case_types)]
            struct #name<#(#ty,)*> {
                #(#children: #children,)*
            }
        };

        for child in &self.children {
            let child = child.struct_def();
            stmt = quote! { #child #stmt };
        }

        stmt
    }

    fn list_init(&self) -> TokenStream {
        let name = &self.name;
        let children_init = self.children.iter().map(|x| x.list_init());
        quote! {
            crate::hacks::GhostNode::new(
                crate::hacks::variadics::l! [
                    #(#children_init,)*
                ],
                &__scope,
                &#name,
            )
        }
    }
}

#[proc_macro]
pub fn ghost_tree(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let Ok(args) =
        syn::punctuated::Punctuated::<Expr, Comma>::parse_separated_nonempty.parse(item.clone())
    else {
        return item;
    };

    if args.is_empty() {
        return quote! { {} }.into();
    }

    let n = args.len() - 1;

    let block = &args[n];

    let tree = &*args
        .iter()
        .take(n)
        .map(Tree::parse_expr)
        .collect::<Vec<_>>();

    let init = tree.iter().map(Tree::init);
    let deinit = tree.iter().map(Tree::deinit);
    let struct_def = tree.iter().map(Tree::struct_def);
    let struct_init = tree.iter().map(Tree::list_init);
    let name = tree.iter().map(|tree| &tree.name);

    let block = quote! {{

        crate::hacks::make_guard!(__scope);
        #(#struct_def)*
        { #(#init)* if const { true } {  #(let #name = #struct_init;)* {#block} } else { #(#deinit)* panic!() } }
    }};

    block.into()
}
