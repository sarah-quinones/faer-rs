# faer/src/serde - Serialization/Deserialization Implementation

## Overview
The `faer/src/serde` module provides serde serialization and deserialization support for faer's matrix types (`Mat`, `MatRef`, `MatMut`).

## File Structure
- **mod.rs**: Module declaration file (declares `mat` submodule)
- **mat.rs**: Core implementation of Serialize/Deserialize traits

## Serialization Format

### Structure
Matrices are serialized as a struct with 3 fields:
```
{
  "nrows": <number>,
  "ncols": <number>,
  "data": [<elements>]
}
```

### Data Layout
- **Row-major order**: Data is serialized by iterating rows first, then columns
- The inner loop iterates over columns (j), outer loop over rows (i)
- For a 3x4 matrix, the order is: (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), ...

### Implementation Details

#### Serialization (faer/src/serde/mat.rs:6-69)
Three implementations:
1. **MatRef<'_, T>** - Primary implementation
   - Uses nested `MatSequenceSerializer` to serialize data as a sequence
   - Serializes as a struct with fields: "nrows", "ncols", "data"

2. **MatMut<'_, T>** - Delegates to MatRef via `self.as_ref().serialize(s)`

3. **Mat<T>** - Delegates to MatRef via `self.as_ref().serialize(s)`

#### Deserialization (faer/src/serde/mat.rs:71-285)
Only implemented for `Mat<T>` (not MatRef or MatMut).

**Key Components:**

1. **Field enum** (lines 78-84): Defines the three expected fields (nrows, ncols, data)

2. **MatVisitor** (line 86): Main visitor struct for deserializing Mat

3. **MatrixOrVec enum** (lines 87-103):
   - Intermediate representation during deserialization
   - Can be either `Matrix(Mat<T>)` or `Vec(alloc::vec::Vec<T>)`
   - Allows flexible deserialization when dimensions are not known upfront
   - `into_mat()` method converts to final Mat

4. **MatrixOrVecDeserializer** (lines 105-201):
   - `DeserializeSeed` implementation
   - Handles two scenarios:
     - **Known dimensions**: Directly deserializes into Mat with proper stride
     - **Unknown dimensions**: Deserializes into Vec first, converts later
   - Validates data length matches nrows * ncols
   - Errors on too few or too many elements

5. **MatVisitor implementations**:
   - **visit_seq** (lines 212-229): Deserializes from sequence format
   - **visit_map** (lines 231-281): Deserializes from map/struct format
     - Handles fields in any order
     - Validates no duplicate fields
     - Ensures all required fields present

## Memory Safety & Performance

### Unsafe Operations
The code uses unsafe operations for performance:
- Direct pointer writes during deserialization (line 173)
- Manual memory management with `set_len(0)` (line 99)
- `set_dims()` call after filling data (line 177)

### Stride Handling
- Uses column stride from Mat during deserialization (line 157)
- Converts flat index to (i,j) coordinates: `(i / ncols, i % ncols)` (line 172)
- Writes to correct memory location: `data.add(i + j * stride)` (line 173)

## Error Handling

### Validation
- **Length validation**: Checks data length matches nrows × ncols
  - Too few elements: Error with actual count vs expected (lines 160-171)
  - Too many elements: Counts extra elements and errors (lines 179-189)
- **Missing fields**: Errors if nrows, ncols, or data missing (lines 273-278)
- **Duplicate fields**: Errors if same field appears twice (lines 240-263)

### Error Messages
- Invalid length: "invalid length {actual}, expected {expected} elements"
- Missing field: Uses serde's standard missing field error
- Invalid sequence: "expected a sequence"
- Invalid type: "expected a faer matrix"

## Test Coverage (faer/src/serde/mat.rs:286-471)

Comprehensive test suite using `serde_test`:

1. **matrix_serialization_normal** (lines 291-322): 3×4 matrix
2. **matrix_serialization_wide** (lines 324-355): 12×1 matrix
3. **matrix_serialization_tall** (lines 357-388): 1×12 matrix
4. **matrix_serialization_zero** (lines 390-409): 0×0 empty matrix
5. **matrix_serialization_errors_too_small** (lines 411-437): Error case with insufficient data
6. **matrix_serialization_errors_too_large** (lines 439-470): Error case with excess data

### Test Matrix Values
Tests use the pattern: `(i + (j * 10)) as f64`
- (0,0) = 0.0, (0,1) = 10.0, (0,2) = 20.0
- (1,0) = 1.0, (1,1) = 11.0, (1,2) = 21.0
- etc.

## Dependencies
- `serde`: Core serialization traits
- `serde::de`: Deserialization infrastructure (DeserializeSeed, SeqAccess, Visitor)
- `serde::ser`: Serialization infrastructure (SerializeSeq, SerializeStruct)
- `serde_test` (test-only): Token-based testing framework
- `crate::internal_prelude::*`: Internal faer types and utilities
- `core::marker::PhantomData`: For type-safe zero-sized markers
- `alloc::vec::Vec` and `alloc::fmt`: No-std compatible allocation

## Key Design Decisions

1. **Row-major serialization**: Matches common matrix representation formats
2. **Owned Mat only for deserialization**: Makes sense as deserializing creates new data
3. **Flexible deserialization**: Accepts both struct and sequence formats
4. **Strict validation**: Ensures data integrity by checking exact length
5. **Generic over element type**: Works with any T that implements Serialize/Deserialize
6. **Struct format with metadata**: Self-describing format includes dimensions

## Usage Example
```rust
use faer::Mat;
use serde_json;

// Serialize
let mat = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
let json = serde_json::to_string(&mat).unwrap();
// {"nrows":2,"ncols":3,"data":[0.0,1.0,2.0,3.0,4.0,5.0]}

// Deserialize
let mat2: Mat<f64> = serde_json::from_str(&json).unwrap();
```
