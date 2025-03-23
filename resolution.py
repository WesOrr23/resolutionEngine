import logging
import timeit


class DisjunctedClause():
    _disjunction_symbol:str = ';'
    _not_symbol:str = '~'
    
    def __init__(self, literal_map:dict[str, 0b0], string_representation:str=None, asserted_literals:0b0=0b0, negated_literals:0b0=0b0, auto_parse:bool=True):
        """
        Initialize a DisjunctedClause object.
        :param literal_map: Dictionary mapping literals to their bit positions.
        :param string_representation: String representation of the clause.
        :param asserted_literals: Binary representation of asserted literals.
        :param negated_literals: Binary representation of negated literals.
        :param auto_parse: Flag to automatically parse the string representation on object initialization.
        """
        # Reference to the literal map which defines how the the final clause representation looks like.
        self.literal_map:dict[str, 0b0] = literal_map
        
        # Initial String Representation
        if string_representation is None: logging.debug("Clause partially initialized with string representation of type 'None.'")
        self.string_representation:str = string_representation
        
        # Binary Representation
        self.asserted_literals:0b0 = asserted_literals
        self.negated_literals:0b0 = negated_literals

        try:
            if auto_parse: self.parse()
        except TypeError:
            logging.error("Failed to parse the clause due to missing string representation.")
        except ValueError:
            logging.error(f"Invalid string representation. Can only include alphanumeric characters and '{DisjunctedClause._disjunction_symbol}' or '{DisjunctedClause._not_symbol}'")

    def parse(self) -> 'DisjunctedClause':
        """
        Parse the string representation of the clause into binary literals.
        
        Each unique literal is assigned a bit in a bit string that represents all possible clauses that can be formed by the present literals in the clause.
        See three examples:
                    ~C  ~B  ~A   C   B   A    (assumed OR between elements)
        "~A;B;C" ->  0   0   1   1   1   0
        "A;~C"   ->  1   0   0   0   0   1
        "B"      ->  0   0   0   0   1   0 

        Asserted literals (non-negated) are stored separately from the negated literals so that if the another unique literal is discovered in a different clause
        later (but in the same resolution process), the representation can be adjusted.

        :return: The parsed DisjunctedClause object.
        """
        if self.string_representation is None: raise TypeError("Expected 'str', got 'None'")    # Throw exception if clause isn't fully initialized.

        # Iterate through each literal in the clause.
        for literal in self.string_representation.split(DisjunctedClause._disjunction_symbol):  
            absolute_literal:str = literal.replace(DisjunctedClause._not_symbol, "").strip()    # Remove the not_symbol from the literal if it is negated (as well as any whitespace).
            if not absolute_literal.isalnum(): raise ValueError                                 # If after removing the not_symbol, there still is any non-alphabetic character in the string, then the string representation is invalid.

            is_negative:bool = len(absolute_literal) != len(literal)                            # Detect this removal if the lengths are different.
            
            if (bit_position := self.literal_map.get(absolute_literal)) is None:                # If the current literal does not yet have an assigned representation,
                map_size:int = len(self.literal_map.keys())                                     # use the current size of the dictionary
                self.literal_map[absolute_literal] = map_size                                   # to assign the literal's bit position in the representation.
                bit_position:int = map_size                                                     # Since the bit position would have been set to None previously, set the value.

            # Set the bit at the previously determined bit-position of the appropriate binary number using bitwise OR and SHIFT.
            if is_negative: self.negated_literals  |= (1 << bit_position)                       
            else:           self.asserted_literals |= (1 << bit_position)
        
        return self

    def copy(self) -> 'DisjunctedClause':
        """
        Create a copy of the current DisjunctedClause object.
        :return: A new DisjunctedClause object with the same properties.
        """
        return DisjunctedClause(literal_map=self.literal_map, string_representation=self.string_representation)

    def negate(self) -> list['DisjunctedClause']:
        """
        Negate the current clause, converting disjuncted literals to conjuncted literals (which is why a list is returned).

        Essentially this function swaps the halves of the clause and for each literal in the result, creates an entirely new clause.
        EX: Original    Split        Swapped      Converted to conjunction
            001110  ->  001 110  ->  110 001  ->  [100000, 010000, 000001]
        
        :return: A list of negated DisjunctedClause objects.
        """
        new_clauses:list['DisjunctedClause'] = []

        # Create a copy as to not affect the original clause.
        asserted_literals:0b0   = self.asserted_literals 
        negated_literals:0b0    = self.negated_literals

        # Convert disjuncted asserted literals to conjuncted negated literals. Performed by iterating through each bit and checking if it is set.
        shift_count:int = 0
        while asserted_literals > 0:                                                                                                        # Repeat until no more bits to check.
            if asserted_literals & 1 == 1:                                                                                                  # If the current bit is 1,
                new_clauses.append(DisjunctedClause(literal_map=self.literal_map, negated_literals=(1 << shift_count), auto_parse=False))   # create a new clause, being sure to NOT parse it (its already in binary form).
            asserted_literals >>= 1                                                                                                         # Shift right.
            shift_count += 1                                                                                                                # Count shifts.

        # Convert disjuncted negated literals to conjuncted asserted literals. Process is the same as above.
        shift_count:int = 0
        while negated_literals > 0:
            if negated_literals & 1 == 1:
                new_clauses.append(DisjunctedClause(literal_map=self.literal_map, asserted_literals=(1 << shift_count), auto_parse=False))
            negated_literals >>= 1
            shift_count += 1
        
        return new_clauses

    def merge_halves(self) -> 0b0:
        """
        Merge asserted and negated literals into a single binary representation.
        :return: Merged binary representation of the clause.
        """
        return (self.negated_literals << len(self.literal_map)) | self.asserted_literals

    def __str__(self):
        """
        String representation of the DisjunctedClause object.

        1. Merge the asserted and negated binary representations.
        2. Convert the resultant number to binary string.
        3. Remove the "0b" prefix temporarily.
        4. Add leading zeros (for consistency with other clauses).
        5. Add back "0b" prefix.

        :return: Binary string representation of the clause.
        """
        map_size = len(self.literal_map.keys())
        return f"0b{bin(self.merge_halves())[2:].zfill(2*map_size)}"
                 #5  #2         #1           #3   #4  -- See above steps.

    def resolve(self, other_clause:'DisjunctedClause') -> 'DisjunctedClause':
        """
        Resolve the current clause with another clause.

        **EX:
        Step 1: Bitwise OR the clauses:
        0000111000
        0110000010
        ------------  OR
        0110111010.... set this aside.

        Step 2: Split the above result in half: 0110111010  ->  01101  11010, XOR these halves:
        01101
        11010
        ----- XOR
        10111... store in bitwise_mask

        Step 3: AND the or_result and bitwise_mask, where the bitwise_mask is duplicated and concatenated:
        0110111010
        1011110111
        ----------- AND
        0010110010... this is the final clause! It works :D.

        Step 4: Split this clause into its asserted and negated parts.
        0010110010  ->  00101  10010

        **The primary advantage to this approach is that it resolves all complementary literals between two clauses at once,
        all within a handful of CPU cycles, as opposed to numerous linear-time string searches, which are several cycles each.

        **Furthermore, it inherently prevents tautologies from arising.

        :param other_clause: The other DisjunctedClause to resolve with.
        :return: The resolved DisjunctedClause object, or None if resolution results in an empty clause.
        """
        start = timeit.default_timer()
        # Get the full binary representation for each clause.
        clause_A = self.merge_halves()
        clause_B = other_clause.merge_halves()
        
        # Bitwise OR the clauses.
        A_or_B = clause_A | clause_B
        
        map_size = len(self.literal_map)     # Will be useful...

        # Generate a "bit-mask" for removing complementary literals. For a clause with 4 literals (map size of 4)...
        right_mask = ~(-1 << map_size)                                          # 00001111  For isolating asserted literals in the OR result.
        left_mask = right_mask << map_size                                      # 11110000  For isolating negated literals in the OR result.
        bit_mask = ((A_or_B & left_mask) >> map_size) ^ (A_or_B & right_mask)   # XOR each half.
        
        # Use the "bit-mask" to determine the final result.
        result = A_or_B & (bit_mask | bit_mask << map_size)
        
        # Split the result into its components
        asserted_literals = result & ((1 << map_size) - 1)                  # Keep only the leftmost <map_size> bits by ANDing
        negated_literals = (result >> map_size) & ((1 << map_size) - 1)     # Keep the the rightmost bits...
        end = timeit.default_timer()

        # Return None if a contradiction (empty clause) is found, otherwise a new clause to be added into the knowledge base!
        return None if result == 0 else DisjunctedClause(literal_map=self.literal_map, asserted_literals=asserted_literals, negated_literals=negated_literals, auto_parse=False)

    def __eq__(self, other):
        """
        Equality check for DisjunctedClause objects. Equal if both asserted_literals and negated_literals are the same.
        :param other: The other DisjunctedClause to compare with.
        :return: True if both clauses are equal, False otherwise.
        """
        if not isinstance(other, DisjunctedClause):
            return NotImplemented
        return self.asserted_literals == other.asserted_literals and self.negated_literals == other.negated_literals

    def __hash__(self):
        """
        Hash function for DisjunctedClause objects.
        :return: Hash value of the clause.
        """
        return hash((self.asserted_literals, self.negated_literals))

def resolution(input_knowledge_base:list[str], query:list[str]):
    """
    Perform resolution on the input knowledge base and query.
    :param input_knowledge_base: List of clauses in the knowledge base.
    :param query: List of query clauses.
    :return: Tuple containing whether the query is entailed and if the knowledge base is unsatisfiable.
    """
    
    def resolve(resolved_pairs:set[tuple[int]]=None):
        """
        Perform the resolution process on the knowledge base.

        The resolution process iteratively attempts to derive new clauses by resolving pairs of existing clauses.
        The process continues until either a contradiction (an empty clause) is derived, indicating that the knowledge base is unsatisfiable,
        or no new clauses can be derived, indicating that the knowledge base is satisfiable.

        Steps:
        1. Initialize a set to keep track of resolved clause pairs and a counter for newly added clauses.
        2. Continuously loop until a contradiction is found or the knowledge base is deemed satisfiable.
        3. For each pair of clauses in the knowledge base:
            a. Skip if the clauses are identical or have been resolved before.
            b. Resolve the pair to derive a new clause (resolvent).
            c. If the resolvent is an empty clause, return True (indicating unsatisfiability).
            d. If the resolvent is new, add it to the set of derived clauses.
        4. If no new clauses were derived in the current iteration, return False (indicating satisfiability).
        5. Update the knowledge base with the newly derived clauses and continue the process.
        
        :param: resolved_pairs, (Optional) If desired, resolved_pairs can begin with values. 
                This is used if resolution has been already ran once on the majority of a KB and adding a new clause(s).
        :return: True if a contradiction is found (unsatisfiable), False otherwise (satisfiable).
        """
        if resolved_pairs is None: resolved_pairs = set()
        no_previous_clauses_added:int = 0
        while True:
            derived_clauses:set[DisjunctedClause] = set()
            for clause_A in converted_knowledge_base[-no_previous_clauses_added:] or converted_knowledge_base:
                for clause_B in converted_knowledge_base:

                    if clause_A.__eq__(clause_B): continue                          # Are the two clauses being compared the same? If so, skip!
                    
                    clause_pair = tuple(sorted([id(clause_A), id(clause_B)]))       # Otherwise, group these two into a pair by a hashID (sorted so that (A, B) is the same as (B, A))
                    if clause_pair in resolved_pairs: continue                      # If this pair has already been resolved, skip!
                    
                    resolvent:DisjunctedClause = clause_A.resolve(clause_B)         # Resolve the two clauses.

                    if resolvent is None: return True, resolved_pairs               # If resolving the clauses results in an empty clause (a contradiction), then YAY, all done.
                    resolved_pairs.add(clause_pair)                                 # Otherwise, add this pair to those that have been resolved (so we don't try to resolve them again).
                    
                    if resolvent not in cKB_set:                                    # If this resolvent is not yet in the knowledge base (checking the hashedSet for constant time time complexity),
                        derived_clauses.add(resolvent)                              # then add it!
            
            if not derived_clauses: return False, resolved_pairs                    # If for every pair of clauses, none of them create a new clause, then the knowledge base is satisfiable (no contradiction can be found).
            
            no_previous_clauses_added:int = len(derived_clauses)                    # Otherwise, note how many clauses were added for optimizing the outer loop of the next iteration. Instead of going through all clauses (in the outer loop), just go through the new ones.
            converted_knowledge_base.extend(derived_clauses)                        # Add the new clauses to the knowledge base.
            cKB_set.update(derived_clauses)

    converted_knowledge_base:list[DisjunctedClause] = []    # Stores all of the binary representations of everything we know (our evidence)!
    cKB_set:set[DisjunctedClause] = set()
    literal_map:dict[str, 0b0] = {}                         # Maps a string literal to its binary representation (bit position).
    
    logging.info("Beginning KB Conversion")

    # Iterate through each clause in the knowledge base and convert these first.
    for i, clause in enumerate(input_knowledge_base, start=1):
        input_KB_clause = DisjunctedClause(literal_map=literal_map, string_representation=clause)
        converted_knowledge_base.append(input_KB_clause)
        cKB_set.add(input_KB_clause)
    
    logging.info("KB Conversion Complete")
    KBstart = timeit.default_timer()

    # Before adding the query, try resolving. If this returns true, that means the knowledge base is unsatisfiable. Any query will be entailed. Also retrieve the resolved_pairs so that when running below, we don't start from scratch!
    unsatisfiable_KB, resolved_pairs = resolve()
    KBend = timeit.default_timer()
    logging.info("KB Resolution Complete")

    # Iterate through each clause in the query.
    for i, clause in enumerate(query, start=1):
        query_clause = DisjunctedClause(literal_map=literal_map, string_representation=clause)
        negated_clauses = query_clause.negate()                                                 # Negate the clause, which may return multiple conjuncted clauses.
        converted_knowledge_base.extend(negated_clauses)
        cKB_set.update(negated_clauses)  # Add each to the knowledge base.

    logging.info("Query Conversion Complete")

    Qstart = timeit.default_timer()
    query_entailed, _ = resolve(resolved_pairs=resolved_pairs)                                  # Use the resolved pairs from the first resolution...
    Qend = timeit.default_timer()
    logging.info("Query Resolution Complete")

    runtime = (KBend - KBstart) + (Qend - Qstart)

    return query_entailed, unsatisfiable_KB, runtime                                            # Return the resolution results!
    
def auto_format_time(value: float) -> str:
    """
    Formats time to the most readable unit dynamically.
    :param value: Time value in seconds.
    :return: Formatted time string.
    """
    units = {
        "s": 1, 
        "ms": 1e-3, 
        "µs": 1e-6, 
        "ns": 1e-9
    }
    
    for unit, threshold in units.items():
        if value >= threshold:
            return f"{value / threshold:07.3f}{unit}"
    
    return f"{value:.3f} ns"  # Default to ns for very small values

def run_test_cases():
    """
    Run predefined test cases to validate the resolution algorithm.
    """
    test_cases = [
        # 1. Contradiction in KB: p and ~p
        (['p', '~p'], ['q'], True),
        
        # 2. Empty KB with a simple query
        ([], ['a'], False),
        
        # 3. Simple entailment: KB directly contains the query
        (['a'], ['a'], True),
        
        # 4. Simple non-entailment: KB doesn’t imply the query
        (['a'], ['b'], False),
        
        # 5. Contradiction with multiple clauses
        (['p', '~p;q', '~q;r', 't'], ['r'], True),
        
        # 6. Empty KB with a complex query
        ([], ['~a;b'], False),
        
        # 7. Chaining: Testing implication chains
        (['~a;b', '~b;c', 'a'], ['c'], True),
        
        # 8. Consistent KB with no entailment
        (['~a;b', 'a'], ['c'], False),
        
        # 9. Tautology in KB
        (['a;~a'], ['b'], False),
        
        # 10. Contradictory query negation (rare case)
        (['a'], ['b;~b'], True),

        # 11. Large KB
        (['~a;b', '~b;c', '~c;d', '~d;e', '~e;f', 'a'], ['f'], True),
    ]
    
    for i, (kb, query, expected) in enumerate(test_cases, start=1):
        query_entailed, unsatisfiable_KB, runtime = resolution(kb, query)
        
        print(f"Test Case #{i:0>2}: KB: {str(kb):<50} Query: {str(query):<15} -- {"Passed" if query_entailed == expected else "Failed"} in {auto_format_time(runtime)}")

        if unsatisfiable_KB: logging.info("Input knowledge base contains a contradiction (is unsatisfiable), thus any query will return as entailed.")
        if len(kb) < 1: logging.info("Input knowledge base is empty, thus there is no evidence to conclude meaninful result. Query will never return as entailed.")


if __name__ == "__main__":
    # Custom Logger
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[36m",    # Cyan
            "INFO": "\033[32m",     # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "CRITICAL": "\033[35m", # Magenta
            "RESET": "\033[0m",     # Reset to default
        }
        
        def format(self, record):
            log_color = ColoredFormatter.COLORS.get(record.levelname, ColoredFormatter.COLORS["RESET"])
            message = super().format(record)
            return f"{log_color}{message}{ColoredFormatter.COLORS['RESET']}"
    class BatchingLogHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.last_message = None
            self.count = 0

        def emit(self, record):
            msg = self.format(record)
            
            if msg == self.last_message:
                self.count += 1
            else:
                if self.last_message is not None and self.count > 1:
                    print(f'{self.count}x{self.last_message}')
                elif self.last_message is not None:
                    print(self.last_message)
                
                self.last_message = msg
                self.count = 1

        def flush(self):
            """Ensure the last message gets printed before shutdown"""
            if self.last_message is not None:
                if self.count > 1:
                    print(f'{self.count}x{self.last_message}')
                else:
                    print(self.last_message)
    
    log = logging.getLogger()
    log_handler = BatchingLogHandler()
    log_formatter = ColoredFormatter("%(levelname)s: %(message)s")
    log_handler.setFormatter(log_formatter)
    log.addHandler(log_handler)
    log.setLevel(logging.ERROR)

    # Run the test-cases, timing them for efficiency.
    iterations = 1
    runtime = timeit.timeit(lambda: run_test_cases(), number=iterations)

    print(f"Average Runtime: {auto_format_time(runtime / iterations)}")
