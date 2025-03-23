# Resolution Algorithm for Propositional Logic

This project implements an efficient **resolution algorithm** for propositional logic in Python. The algorithm determines whether a query is entailed by a given knowledge base (KB) by leveraging **bitwise operations** for clause representation and resolution, ensuring optimized performance and scalability. The project is designed using **object-oriented principles**, making the code modular, maintainable, and easy to extend.

## Key Features
- **Bitwise Clause Representation**: Clauses are represented using bitwise operations for efficient manipulation and resolution.
- **Object-Oriented Design**: The project is structured using classes (e.g., `DisjunctedClause`), promoting code reusability and clarity.
- **Efficient Resolution**: The resolution process is optimized to handle contradictions and derive new clauses efficiently.
- **Test Suite**: Includes a comprehensive set of test cases to validate correctness across various scenarios, including edge cases.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/resolution-algorithm.git
   cd resolution-algorithm
Install Dependencies:
Python 3.x is required.
No additional libraries are needed beyond the standard library.
Usage
Running the Algorithm:
The main script (resolution.py) contains the resolution algorithm and test cases.
To run the test cases, execute:
bash

## How It Works
- Clause Representation: Each clause is represented as a DisjunctedClause object, where literals are mapped to bit positions for efficient manipulation.
Resolution Process: The algorithm iteratively resolves pairs of clauses, using bitwise operations to detect and remove all complementary literals between two clauses in one go.
- Contradiction Detection: If an empty clause is derived during resolution, it indicates a contradiction, proving the query is entailed.

## Testing
The project includes a suite of test cases to verify correctness, covering scenarios such as:
- Contradictory knowledge bases
- Simple entailment and non-entailment
- Chaining implications
- Tautologies and contradictions in queries

## Performance
The use of bitwise operations significantly reduces the time complexity of clause resolution, making the algorithm efficient for larger inputs. At the time of upload (to be updated), there is a current issue that leads these gains to be unrealized. I am currently experimenting with a C++ verison of this project. 

## Future Improvements
- Support for Larger Inputs: Extend the bitwise representation to handle more than 32 literals using libraries like gmpy2.
- Parallel Processing: Implement parallel resolution to speed up the process for large knowledge bases.
- GUI Integration: Develop a graphical interface for visualizing clause resolution steps.
