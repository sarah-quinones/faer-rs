use std::collections::HashSet;

use faer_core::{AsMatRef, ComplexField, Mat, RealField};
use proptest::{
    prelude::Rng,
    strategy::{NewTree, Strategy, ValueTree},
    test_runner::{Reason, TestRunner},
};

use crate::Faer;

pub fn mat<E: Strategy>(element: E) -> MatStrategy<E> {
    mat_with(element, Parameters::default())
}

pub fn mat_with<E: Strategy>(element: E, parameters: Parameters) -> MatStrategy<E> {
    MatStrategy {
        element,
        parameters,
    }
}

pub fn square_mat<E: Strategy>(element: E) -> MatStrategy<E> {
    square_mat_with(element, Parameters::default())
}

pub fn square_mat_with<E: Strategy>(element: E, parameters: Parameters) -> MatStrategy<E> {
    MatStrategy {
        element,
        parameters: parameters.square(),
    }
}

#[allow(unused)]
pub fn structure() -> MatrixStructure {
    MatrixStructure::default()
}

#[derive(Debug, Clone, Copy)]
#[allow(unused)]
pub enum Triangular {
    Lower,
    Upper,
    StrictLower,
    StrictUpper,
}

#[derive(Debug, Clone, Default)]
pub struct MatrixStructure {
    square: bool,
    triangular: Option<Triangular>,
    unit_diagonal: bool,
}

impl MatrixStructure {
    pub fn square(self) -> Self {
        Self {
            square: true,
            ..self
        }
    }

    pub fn triangular(self, triangular: Triangular) -> Self {
        Self {
            triangular: Some(triangular),
            ..self
        }
    }

    pub fn unit_diagonal(self) -> Self {
        Self {
            unit_diagonal: true,
            ..self
        }
    }

    pub fn unit_triangular(self, triangular: Triangular) -> Self {
        self.triangular(triangular).unit_diagonal()
    }

    fn is_triangular_or_square(&self) -> bool {
        self.triangular.is_some() || self.square
    }
}

#[derive(Debug, Clone)]
pub struct Parameters {
    max_dim: usize,
    structure: MatrixStructure,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            max_dim: 32,
            structure: MatrixStructure::default(),
        }
    }
}

#[allow(unused)]
impl Parameters {
    pub fn max_dim(self, max_dim: usize) -> Self {
        Self { max_dim, ..self }
    }

    pub fn structure(self, structure: MatrixStructure) -> Self {
        Self { structure, ..self }
    }

    pub fn square(self) -> Self {
        Self {
            structure: self.structure.square(),
            ..self
        }
    }

    pub fn triangular(self, triangular: Triangular) -> Self {
        Self {
            structure: self.structure.triangular(triangular),
            ..self
        }
    }

    pub fn unit_triangular(self, triangular: Triangular) -> Self {
        Self {
            structure: self.structure.unit_triangular(triangular),
            ..self
        }
    }

    pub fn unit_diagonal(self) -> Self {
        Self {
            structure: self.structure.unit_diagonal(),
            ..self
        }
    }
}

#[derive(Debug)]
pub struct MatStrategy<E> {
    element: E,
    parameters: Parameters,
}

impl<E: Strategy> Strategy for MatStrategy<E>
where
    E::Value: ComplexField,
{
    type Tree = MatValueTree<E::Tree>;
    type Value = Mat<E::Value>;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let ncols = runner.rng().gen_range(0..=self.parameters.max_dim);

        if ncols == 0 {
            return Ok(MatValueTree::new(State::empty(), self.parameters.clone()));
        }

        let nrows = if self.parameters.structure.square {
            ncols
        } else {
            runner.rng().gen_range(1..=self.parameters.max_dim)
        };

        Ok(MatValueTree::new(
            State::new(nrows, ncols, &self.parameters, &self.element, runner)?,
            self.parameters.clone(),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
enum Shrink {
    // Remove given row.
    RemoveRow(usize),
    // Remove given column.
    RemoveCol(usize),
    // Remove given row and column simultaneously.
    RemoveRowCol(usize, usize),
    // Shrink the element at given index (in linearized matrix form).
    Element(usize),
}

#[derive(Debug)]
pub struct MatValueTree<E: ValueTree> {
    shrink: Shrink,
    previous: Option<Shrink>,
    state: State<E>,
    parameters: Parameters,
}

impl<E: ValueTree> MatValueTree<E> {
    fn new(state: State<E>, parameters: Parameters) -> Self {
        let shrink = if parameters.structure.is_triangular_or_square() {
            // We must respect the special structure of triangular and square
            // matrices during shrinking.
            Shrink::RemoveRowCol(0, 0)
        } else {
            // Otherwise, we start with removing rows.
            Shrink::RemoveRow(0)
        };

        Self {
            shrink,
            previous: None,
            state,
            parameters,
        }
    }
}

impl<E: ValueTree> ValueTree for MatValueTree<E>
where
    E::Value: ComplexField,
{
    type Value = Mat<E::Value>;

    fn current(&self) -> Self::Value {
        self.state.current()
    }

    fn simplify(&mut self) -> bool {
        let (nrows, ncols) = self.state.dims();

        loop {
            match self.shrink {
                Shrink::RemoveRow(row) => {
                    if row == nrows {
                        self.shrink = if self.parameters.structure.is_triangular_or_square() {
                            // We must respect the special structure of
                            // triangular and square matrices and thus go
                            // straight to shrinking the elements.
                            Shrink::Element(0)
                        } else {
                            // Otherwise, we continue with removing columns.
                            Shrink::RemoveCol(0)
                        };
                    } else {
                        self.previous = Some(self.shrink);
                        self.state.remove_row(row);
                        self.shrink = Shrink::RemoveRow(row + 1);
                        return true;
                    }
                }
                Shrink::RemoveCol(col) => {
                    if col == ncols {
                        self.shrink = Shrink::Element(0);
                    } else {
                        self.previous = Some(self.shrink);
                        self.state.remove_col(col);
                        self.shrink = Shrink::RemoveCol(col + 1);
                        return true;
                    }
                }
                Shrink::RemoveRowCol(row, col) => {
                    if row == nrows || col == ncols {
                        self.shrink = match nrows.cmp(&ncols) {
                            // Square matrix, continue with shrinking elements.
                            core::cmp::Ordering::Equal => Shrink::Element(0),
                            // Rectangular matrix with more rows than columns,
                            // continue with removing rows beyond the square.
                            core::cmp::Ordering::Greater => Shrink::RemoveRow(row + 1),
                            // Rectangular matrix with more columns than rows,
                            // continue with removing columns beyond the square.
                            core::cmp::Ordering::Less => Shrink::RemoveCol(col + 1),
                        };
                    } else {
                        self.previous = Some(self.shrink);
                        self.state.remove_row(row);
                        self.state.remove_col(col);
                        self.shrink = Shrink::RemoveRowCol(row + 1, col + 1);
                        return true;
                    }
                }
                Shrink::Element(i) => {
                    let (row, col) = row_col(i, ncols);

                    if i == self.state.len() {
                        return false;
                    } else if !self.state.contains(row, col) {
                        self.shrink = Shrink::Element(i + 1);
                    } else if self.state.simplify(i) {
                        self.previous = Some(self.shrink);
                        return true;
                    } else {
                        self.shrink = Shrink::Element(i + 1);
                    }
                }
            }
        }
    }

    fn complicate(&mut self) -> bool {
        // `self.previous.take()` gives me Rust-Analyzer type error, although
        // cargo check is fine.
        match std::mem::take(&mut self.previous) {
            None => false,
            Some(Shrink::RemoveRow(row)) => {
                self.state.restore_row(row);
                true
            }
            Some(Shrink::RemoveCol(col)) => {
                self.state.restore_col(col);
                true
            }
            Some(Shrink::RemoveRowCol(row, col)) => {
                self.state.restore_row(row);
                self.state.restore_col(col);
                true
            }
            Some(Shrink::Element(i)) => {
                if self.state.complicate(i) {
                    self.previous = Some(Shrink::Element(i));
                    true
                } else {
                    false
                }
            }
        }
    }
}

fn row_col(index: usize, ncols: usize) -> (usize, usize) {
    (index / ncols, index % ncols)
}

#[derive(Debug)]
struct State<E: ValueTree> {
    ncols: usize,
    elements: Vec<Element<E, E::Value>>,
    removed_rows: HashSet<usize>,
    removed_cols: HashSet<usize>,
}

impl<E: ValueTree> State<E>
where
    E::Value: ComplexField,
{
    fn empty() -> Self {
        State {
            ncols: 0,
            elements: Vec::new(),
            removed_rows: HashSet::new(),
            removed_cols: HashSet::new(),
        }
    }

    fn new<S>(
        nrows: usize,
        ncols: usize,
        parameters: &Parameters,
        element: &S,
        runner: &mut TestRunner,
    ) -> Result<Self, Reason>
    where
        S: Strategy<Tree = E, Value = E::Value>,
    {
        let len = nrows * ncols;
        let elements = (0..len)
            .map(|i| {
                let (row, col) = row_col(i, ncols);

                if should_generate(&parameters.structure, row, col) {
                    if let Some(forced) = forced_value::<E::Value>(&parameters.structure, row, col)
                    {
                        Ok(Element::Just(forced))
                    } else {
                        element.new_tree(runner).map(Element::Random)
                    }
                } else {
                    Ok(Element::Just(E::Value::faer_zero()))
                }
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            ncols,
            elements,
            removed_rows: HashSet::new(),
            removed_cols: HashSet::new(),
        })
    }

    fn dims(&self) -> (usize, usize) {
        (self.elements.len() / self.ncols, self.ncols)
    }

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn contains(&self, row: usize, col: usize) -> bool {
        !(self.removed_rows.contains(&row) || self.removed_cols.contains(&col))
    }

    fn current(&self) -> Mat<E::Value> {
        if self.elements.is_empty() {
            return Mat::new();
        }

        let elements = self.elements.iter().enumerate().filter_map(|(i, e)| {
            let (row, col) = row_col(i, self.ncols);

            if self.removed_rows.contains(&row) || self.removed_cols.contains(&col) {
                None
            } else {
                Some(e.current())
            }
        });

        let ncols = self.ncols - self.removed_cols.len();
        let nrows = self.elements.len() / self.ncols - self.removed_rows.len();

        let mut mat = Mat::zeros(nrows, ncols);

        for (i, e) in elements.enumerate() {
            let (row, col) = row_col(i, ncols);
            mat.write(row, col, e);
        }

        mat
    }

    fn remove_row(&mut self, row: usize) {
        self.removed_rows.insert(row);
    }

    fn remove_col(&mut self, col: usize) {
        self.removed_cols.insert(col);
    }

    fn restore_row(&mut self, row: usize) {
        self.removed_rows.remove(&row);
    }

    fn restore_col(&mut self, col: usize) {
        self.removed_cols.remove(&col);
    }

    fn simplify(&mut self, elem: usize) -> bool {
        self.elements[elem].simplify()
    }

    fn complicate(&mut self, elem: usize) -> bool {
        self.elements[elem].complicate()
    }
}

fn should_generate(structure: &MatrixStructure, row: usize, col: usize) -> bool {
    if let Some(triangular) = structure.triangular {
        match triangular {
            Triangular::Lower => row >= col,
            Triangular::Upper => row <= col,
            Triangular::StrictLower => row > col,
            Triangular::StrictUpper => row < col,
        }
    } else {
        true
    }
}

fn forced_value<E: ComplexField>(structure: &MatrixStructure, row: usize, col: usize) -> Option<E> {
    if structure.unit_diagonal && row == col {
        Some(E::faer_one())
    } else {
        None
    }
}

#[derive(Debug)]
enum Element<T, E> {
    Random(T),
    Just(E),
}

impl<T, E> Strategy for Element<T, E>
where
    T: Strategy<Value = E>,
    E: Clone + core::fmt::Debug,
{
    type Tree = Element<T::Tree, E>;
    type Value = E;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        match self {
            Element::Random(strategy) => strategy.new_tree(runner).map(Element::Random),
            Element::Just(value) => Ok(Element::Just(value.clone())),
        }
    }
}

impl<T, E> ValueTree for Element<T, E>
where
    T: ValueTree<Value = E>,
    E: Clone + core::fmt::Debug,
{
    type Value = E;

    fn current(&self) -> Self::Value {
        match self {
            Element::Random(tree) => tree.current(),
            Element::Just(value) => value.clone(),
        }
    }

    fn simplify(&mut self) -> bool {
        match self {
            Element::Random(tree) => tree.simplify(),
            Element::Just(_) => false,
        }
    }

    fn complicate(&mut self) -> bool {
        match self {
            Element::Random(tree) => tree.complicate(),
            Element::Just(_) => false,
        }
    }
}

pub fn relative_epsilon_norms<E>(
    x: impl AsMatRef<E>,
    y: impl AsMatRef<E>,
    threshold: E::Real,
) -> E::Real
where
    E: ComplexField,
{
    let x = x.as_mat_ref();
    let y = y.as_mat_ref();

    let dim_max = x.nrows().max(x.ncols()).max(y.nrows()).max(y.ncols());
    let dim_max = E::Real::faer_from_f64(dim_max as f64);

    let x_norm = x.norm_max();
    let y_norm = y.norm_max();
    let norm_max = if x_norm > y_norm { x_norm } else { y_norm };

    threshold
        .faer_mul(E::Real::faer_epsilon().unwrap())
        .faer_mul(dim_max)
        .faer_mul(norm_max)
}

#[allow(unused)]
pub fn relative_epsilon_cond<E>(
    a: impl AsMatRef<E>,
    b: impl AsMatRef<E>,
    threshold: E::Real,
) -> E::Real
where
    E: ComplexField,
{
    let a = a.as_mat_ref();
    let b = b.as_mat_ref();

    let nrows = a.nrows();
    if nrows == 0 {
        return E::Real::faer_zero();
    }

    let (sing_min, sing_max) = a.singular_values().into_iter().fold(
        (E::Real::faer_from_f64(f64::INFINITY), E::Real::faer_zero()),
        |(sing_min, sing_max), sing| {
            let sing_min = if sing < sing_min { sing } else { sing_min };
            let sing_max = if sing > sing_max { sing } else { sing_max };
            (sing_min, sing_max)
        },
    );

    let a_cond = sing_max.faer_div(sing_min);
    let b_norm = b.norm_l2();

    let eps = E::Real::faer_epsilon().unwrap();

    eps.faer_mul(threshold.faer_add(threshold.faer_mul(a_cond)))
        .faer_mul(b_norm)
}
