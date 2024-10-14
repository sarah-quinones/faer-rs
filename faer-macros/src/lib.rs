use proc_macro2::TokenStream;
use quote::quote;
use std::iter;
use syn::{
    parse::Parser,
    punctuated::Punctuated,
    spanned::Spanned,
    token::Comma,
    visit_mut::{self, VisitMut},
    Expr, ExprCall, ExprMethodCall, ExprPath, ExprReference, Ident, Macro, Path, PathSegment,
};

struct RustCtx;
struct MathCtx<'a> {
    ctx: &'a Expr,
    real_ctx: &'a Expr,
    real: bool,
}

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

impl visit_mut::VisitMut for MathCtx<'_> {
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
        match i {
            Expr::Index(idx) => {
                let expr = &mut *idx.expr;
                *i = Expr::MethodCall(ExprMethodCall {
                    attrs: vec![],
                    receiver: Box::new(Expr::MethodCall(ExprMethodCall {
                        attrs: vec![],
                        receiver: Box::new(expr.clone()),
                        dot_token: Default::default(),
                        method: Ident::new("rb", idx.index.span()),
                        turbofish: None,
                        paren_token: Default::default(),
                        args: std::iter::empty::<Expr>().collect(),
                    })),
                    dot_token: Default::default(),
                    method: Ident::new("__at", idx.index.span()),
                    turbofish: None,
                    paren_token: Default::default(),
                    args: iter::once((*idx.index).clone()).collect(),
                });

                return;
            }
            _ => {}
        }

        visit_mut::visit_expr_mut(self, i);

        match i {
            Expr::Unary(unary) => match unary.op {
                syn::UnOp::Neg(minus) => {
                    *i = Expr::Call(ExprCall {
                        attrs: vec![],
                        func: Box::new(ident_expr(&Ident::new("neg", minus.span))),
                        paren_token: Default::default(),
                        args: std::iter::once((*unary.expr).clone()).collect(),
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
                    syn::BinOp::Eq(eq_eq) => Some(Ident::new("eq", eq_eq.span())),
                    syn::BinOp::Ne(ne) => Some(Ident::new("ne", ne.span())),
                    syn::BinOp::Le(le) => Some(Ident::new("le", le.span())),
                    syn::BinOp::Ge(ge) => Some(Ident::new("ge", ge.span())),
                    syn::BinOp::Lt(lt) => Some(Ident::new("lt", lt.span)),
                    syn::BinOp::Gt(gt) => Some(Ident::new("gt", gt.span)),
                    _ => None,
                };
                if let Some(func) = func {
                    *i = Expr::Call(ExprCall {
                        attrs: vec![],
                        func: Box::new(ident_expr(&func)),
                        paren_token: Default::default(),
                        args: [(*binop.left).clone(), (*binop.right).clone()]
                            .into_iter()
                            .collect(),
                    })
                }
            }
            _ => {}
        }

        match i {
            Expr::Call(call) => match &*call.func {
                Expr::Path(path) => {
                    if let Some(method) = path.path.get_ident() {
                        let span = method.span();

                        *i = math_expr(
                            span,
                            if self.real { self.real_ctx } else { self.ctx }.clone(),
                            method,
                            call.args.iter(),
                        );
                    }
                }
                _ => {}
            },
            Expr::MethodCall(call) => match &*call.receiver {
                Expr::Path(path)
                    if path
                        .path
                        .get_ident()
                        .map(|ident| ident.to_string())
                        .as_deref()
                        == Some("re") =>
                {
                    let method = &call.method;
                    let span = method.span();

                    *i = math_expr(span, self.real_ctx.clone(), method, call.args.iter());
                }
                Expr::Path(path)
                    if path
                        .path
                        .get_ident()
                        .map(|ident| ident.to_string())
                        .as_deref()
                        == Some("cx") =>
                {
                    let method = &call.method;
                    let span = method.span();

                    *i = math_expr(span, self.ctx.clone(), method, call.args.iter());
                }
                _ => {}
            },
            _ => {}
        }
    }
}

impl visit_mut::VisitMut for RustCtx {
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
        match i {
            Expr::MethodCall(call) if call.method.to_string().as_str() == "re" => {
                match &*call.receiver {
                    Expr::Path(path)
                        if path
                            .path
                            .get_ident()
                            .map(|name| name.to_string())
                            .as_deref()
                            == Some("math") =>
                    {
                        let first = call.args.first_mut().unwrap();
                        let span = call.method.span();
                        let ctx = &syn::Ident::new("ctx", span);
                        MathCtx {
                            ctx: &ident_expr(&syn::Ident::new("ctx", call.method.span())),
                            real_ctx: &Expr::MethodCall(ExprMethodCall {
                                attrs: vec![],
                                receiver: Box::new(ident_expr(ctx)),
                                dot_token: Default::default(),
                                method: Ident::new("real_ctx", span),
                                turbofish: None,
                                paren_token: Default::default(),
                                args: Punctuated::new(),
                            }),
                            real: true,
                        }
                        .visit_expr_mut(first);
                        *i = first.clone();
                        return;
                    }
                    _ => {}
                }
            }

            Expr::Call(call) => match &*call.func {
                Expr::Path(path)
                    if path
                        .path
                        .get_ident()
                        .map(|name| name.to_string())
                        .as_deref()
                        == Some("math") =>
                {
                    let first = call.args.first_mut().unwrap();
                    let span = call.func.span();
                    let ctx = &syn::Ident::new("ctx", span);
                    MathCtx {
                        ctx: &ident_expr(&syn::Ident::new("ctx", call.func.span())),
                        real_ctx: &Expr::MethodCall(ExprMethodCall {
                            attrs: vec![],
                            receiver: Box::new(ident_expr(ctx)),
                            dot_token: Default::default(),
                            method: Ident::new("real_ctx", span),
                            turbofish: None,
                            paren_token: Default::default(),
                            args: Punctuated::new(),
                        }),
                        real: false,
                    }
                    .visit_expr_mut(first);
                    *i = first.clone();
                    return;
                }
                _ => {}
            },
            _ => {}
        }

        visit_mut::visit_expr_mut(self, i);

        match i {
            Expr::MethodCall(call) => match &*call.receiver {
                Expr::Field(field) => match &*field.base {
                    Expr::Path(path)
                        if path
                            .path
                            .get_ident()
                            .map(|name| name.to_string())
                            .as_deref()
                            == Some("math") =>
                    {
                        match &field.member {
                            syn::Member::Named(ident) if &*ident.to_string() == "re" => {
                                let method = &call.method;
                                let span = method.span();
                                let ctx = &syn::Ident::new("ctx", span);
                                let receiver = Expr::MethodCall(ExprMethodCall {
                                    attrs: vec![],
                                    receiver: Box::new(ident_expr(ctx)),
                                    dot_token: Default::default(),
                                    method: Ident::new("real_ctx", span),
                                    turbofish: None,
                                    paren_token: Default::default(),
                                    args: Punctuated::new(),
                                });

                                *i = math_expr(span, receiver, method, call.args.iter());
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                },

                Expr::Path(path)
                    if path
                        .path
                        .get_ident()
                        .map(|name| name.to_string())
                        .as_deref()
                        == Some("math") =>
                {
                    let method = &call.method;
                    let span = method.span();
                    let receiver = &syn::Ident::new("ctx", span);

                    *i = math_expr(span, ident_expr(receiver), method, call.args.iter());
                }
                _ => {}
            },
            _ => {}
        };
    }
}

fn math_expr<'a>(
    _: proc_macro2::Span,
    receiver: Expr,
    method: &Ident,
    args: impl Iterator<Item = &'a Expr>,
) -> Expr {
    Expr::MethodCall(ExprMethodCall {
        attrs: vec![],
        receiver: Box::new(receiver),
        dot_token: Default::default(),
        method: method.clone(),
        turbofish: None,
        paren_token: Default::default(),
        args: args
            .cloned()
            .map(|expr| {
                Expr::Reference(ExprReference {
                    attrs: vec![],
                    and_token: Default::default(),
                    mutability: None,
                    expr: Box::new(expr),
                })
            })
            .collect(),
    })
}

#[proc_macro_attribute]
pub fn math(_: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let Ok(mut item) = syn::parse::<syn::ItemFn>(item.clone()) else {
        return item;
    };
    let mut rust_ctx = RustCtx;
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
                crate::hacks::variadics::list! [
                    #(#children_init,)*
                ],
                &__scope,
                &#name,
            )
        }
    }

    fn struct_init(&self) -> TokenStream {
        let name = &self.name;
        let children = self.children.iter().map(|x| &x.name);
        let children_init = self.children.iter().map(|x| x.struct_init());
        quote! {
            crate::hacks::GhostNode::new(
                #name {
                    #(#children: #children_init,)*
                },
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
    let struct_init = tree.iter().map(Tree::struct_init);
    let name = tree.iter().map(|tree| &tree.name);

    let block = quote! {{

        crate::hacks::make_guard!(__scope);
        #(#struct_def)*
        { #(#init)* if const { true } {  #(let #name = #struct_init;)* {#block} } else { #(#deinit)* panic!() } }
    }};

    block.into()
}

#[proc_macro]
pub fn ghost_tree2(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
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
