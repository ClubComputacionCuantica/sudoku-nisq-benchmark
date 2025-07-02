# M. Weiss' implementation to represent a sudoku board

class Board:
    """
    A class to represent a sudoku grid.
    
    Attributes
    ----------
    board_state : [[int]]
        a two dimensional list of ints representing the board
    unit_heigth : int
        the heigth of a sudoku subunit starting at 1..n
    unit_wifth : int
        the width of a sudoku subunit starting at 1..n
    grid_heigth : int
        the number of vertically stacked subunits
    grid_width : int
        the number of laterally stacked subunits
    open_tuples : [((int, int), (int, int))]
        a list of all single field pairs that are not allowed to be the same value
        this list is the minimum subset of all possible field pairs which need to be checked
        to achieve a valid solution
    
    Methods
    -------
    get_unit_size() -> (int, int):
        returns the size of the unit as a tuple (unit_height, unit_width)
    get_grid_size():
        returns the size of the grid as a tuple (grid_height, grid_width)
    get_open_tuples():
        a list of all single field pairs that are not allowed to be the same value
        this list is the minimum subset of all possible field pairs which need to be checked
        to achieve a valid solution
        returns open tuples in format ((vert, lat), (vert, lat))
    get_open_indexed_tuples():
        returns open tuples in format (index, index)
    update_board(values: list of ints, positions: list of tuples):
        sets values in the grid, positions are given as a (height, width) tuple of the grid
    print_board():
        printing the board_state to the standard output this is mainly for testing during development
    """

    def __init__(self, unit_height, unit_width, grid_height, grid_width, init_value):
        """
        Constructs the necessary attributes for the Board object.

        Parameters
        ----------
        unit_heigth : int
            the heigth of a sudoku subunit starting at 1..n
        unit_wifth : int
            the width of a sudoku subunit starting at 1..n
        grid_heigth : int
            the number of vertically stacked subunits starting at 1..n
        grid_width : int
            the number of laterally stacked subunits starting at 1..n
        """
        self.unit_height = unit_height
        self.unit_width = unit_width
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.width_in_fields = unit_width*grid_width
        self.height_in_fields = unit_height*grid_height

        self.__set_board(init_value)

    def __set_board(self, init_value: int):
        """
        Initialises the board_state based on units and grid sizing.
        Acts as a helper function of the __init__ and is private.

        Paraters
        --------
        init_value : int
            The value all fields are initialized with. Usually it is -1 to be empty.
        """

        self.board_state = [[init_value for _ in range(self.width_in_fields)] for _ in range(self.height_in_fields)]

    def get_unit_size(self) -> tuple:
        """
        Returns the size of the unit in an tuple (unit_height, unit_width)

        Returns
        -------
        (int, int)
        """
        return (self.unit_height, self.unit_width)

    def get_grid_size(self) -> tuple:
        """
        Returns the size of the grid in numbers of units as a tuple (grid_height, grid_width)

        Returns
        -------
        (int, int)
        """
        return (self.grid_height, self.grid_width)

    def get_open_tuples(self) -> list:
        """
        Finds all the pairs of two single fields that need to be different for the solution to be valid.
        The pair is defined by a tuple of two fields where each field is again a tuple (int, int).
        The tuple representing a field consists of (height index, width index).

        Returns
        -------
        [((int, int), (int, int))]
        """
        # i and j iterate the board state
        # for every field all pairs in lateral direction all > j are checked
        # in vertical directions all > i are checked
        # for every field also the other field in the subunit are checked
        # the lateral and vertical fields are not checked again
        # if the current filed is the upper left field of a subunit we check all subunit pairs
        # if at least one of the fields has value -1 the pair is added to the return list
        re_list = []

        for vert_index in range(self.height_in_fields):
            for lat_index in range(self.width_in_fields):
                current_field = self.board_state[vert_index][lat_index]
                # vertical pairs
                for vert_sub_index in range(vert_index+1, self.height_in_fields):
                    sub_field = self.board_state[vert_sub_index][lat_index]
                    if min(current_field, sub_field) == -1:
                        re_list.append(((vert_index, lat_index), 
                                        (vert_sub_index, lat_index)))
                # lateral pairs
                for lat_sub_index in range(lat_index+1, self.width_in_fields):
                    sub_field = self.board_state[vert_index][lat_sub_index]
                    if min(current_field, sub_field) == -1:
                        re_list.append(((vert_index, lat_index),
                                        (vert_index, lat_sub_index)))
                # subunit pairs
                vert_sub_start = vert_index-vert_index%self.unit_height
                lat_sub_start = lat_index-lat_index%self.unit_width
                for vert_sub_index in range(vert_sub_start, vert_sub_start+self.unit_height):
                    for lat_sub_index in range(lat_sub_start, lat_sub_start+self.unit_width):
                        if vert_index != vert_sub_index and lat_index != lat_sub_index:
                            if vert_sub_index > vert_index:
                                sub_field = self.board_state[vert_sub_index][lat_sub_index]
                                if min(current_field, sub_field) == -1:
                                    re_list.append(((vert_index, lat_index), 
                                                    (vert_sub_index, lat_sub_index)))

        return re_list

    def get_open_indexed_tuples(self):
        """
        Translates the open tuples from tuples of tuples to tuples of indexes.
        Returns a list of tuples of indexes.
        See readme for explaination on indexes.

        Returns
        -------
        [(int, int)]
        """
        # index dicts has the tuples as keys and the indexes as values
        index_dict = {(vert, lat):(vert*self.width_in_fields+lat) 
                      for vert in range(self.height_in_fields)
                      for lat in range(self.width_in_fields)}

        open_tuples = self.get_open_tuples()
        re_tuples = []

        for tup in open_tuples:
            re_tuples.append((index_dict[tup[0]], index_dict[tup[1]]))
        
        return re_tuples

    def update_board(self, values, positions):
        """
        Updates the board with new values given for a specific position.
        After calling this you should request the new open tuples by get_open_tuples().

        Parameters
        ----------
        values : list of ints
            the values the updates fields will have
        positions : list of tuples
            the position for each value to be updated as a tuple of ints 
            that represent height index and width index of the Board

        Returns
        -------
        None
        """
        for val, pos in zip(values, positions):
            try:
                self.board_state[pos[0]][pos[1]] = val
            except IndexError:
                print(f"The index {pos} is out of range. {val} could not be set to this field.")

    def print_board(self):
        """
        Prints the board_state to the standard output.
        This is mainly for testing during development.
        """
        for row in self.board_state:
            print(row)