import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Integer, Column
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, Any, Optional, List
import warnings
import logging
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
from qiskit_ibm_runtime import QiskitRuntimeService
from pytket.extensions.qiskit import IBMQBackend

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

from sudoku_nisq.q_sudoku import Sudoku
from sudoku_nisq.quantum import ExactCoverQuantumSolver

# ------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------
# Configure logging to capture info and error messages in a log file.
logging.basicConfig(
    filename='app.log',          # Log file name
    filemode='a',                # Append mode (use 'w' to overwrite on each run)
    level=logging.INFO,          # Set the logging level to capture INFO and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Log message format with timestamps
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# SQLAlchemy Setup for Database Interaction
# ------------------------------------------------------------
# Create the base class for declarative class definitions.
Base = declarative_base()

# SQLAlchemy model for representing the 'quantum_resources' table in the database
class QuantumResources(Base):
    """
    SQLAlchemy model representing the 'quantum_resources' table.
    
    Attributes:
        id: Primary key of the record.
        missing_cells: Number of empty cells in the Sudoku puzzle.
        num_qubits_simple_encoding: Number of qubits required using the simple encoding.
        mcx_gates_simple_encoding: Number of MCX gates required using the simple encoding.
        total_gates_simple_encoding: Total number of gates required using the simple encoding.
        num_qubits_pattern_encoding: Number of qubits required using the pattern encoding.
        mcx_gates_pattern_encoding: Number of MCX gates required using the pattern encoding.
        total_gates_pattern_encoding: Total number of gates required using the pattern encoding.
    """
    __tablename__ = 'quantum_resources'

    id = Column(Integer, primary_key=True)
    
    missing_cells = Column(Integer)
    
    num_qubits_simple_encoding = Column(Integer)
    mcx_gates_simple_encoding = Column(Integer)
    total_gates_simple_encoding = Column(Integer)
    
    num_qubits_pattern_encoding = Column(Integer)
    mcx_gates_pattern_encoding = Column(Integer)
    total_gates_pattern_encoding = Column(Integer)

# ------------------------------------------------------------
# Data Generation Class
# ------------------------------------------------------------
class GenData:
    """
    Class for generating Sudoku puzzles and estimating the quantum resources required for solving them.

    Attributes:
        size: Grid size (e.g., 2 for a 4x4 Sudoku).
        board_size: Total number of sub-grids (size * size).
        num_puzzles: Total number of puzzles to generate.
        db_url: Database connection string (optional).
        total_cells: Total number of cells in the Sudoku board.
        data_list: List that stores computed quantum resource data for each puzzle.
        engine: SQLAlchemy engine for database operations (if db_url is provided).
    """
    def __init__(self, grid_size: int = 2, num_puzzles: int = 50, db_url: Optional[str] = None):
        self.size = grid_size
        self.board_size = grid_size * grid_size
        self.num_puzzles = num_puzzles
        self.db_url = db_url
        self.total_cells = self.board_size * self.board_size
        self.data_list: List[Dict[str, Any]] = []
        # If a database URL is provided, create an engine for database interactions.
        if db_url:
            self.engine = create_engine(self.db_url)
        else:
            self.engine = None

    def find_quantum_resources(self, sudoku: Sudoku):
        """
        Compute the quantum resources required for solving a given Sudoku puzzle.

        The method calculates resources using two encoding schemes:
            - Simple encoding
            - Pattern encoding

        Parameters:
            sudoku: A Sudoku instance representing the puzzle.

        Returns:
            A tuple containing:
                sim_num_qubits, sim_mcx_gates, sim_total_gates,
                pat_num_qubits, pat_mcx_gates, pat_total_gates
        """
        # Compute quantum resources for simple encoding
        qsolver = ExactCoverQuantumSolver(sudoku, simple=True, pattern=False)
        sim_num_qubits, sim_mcx_gates, sim_total_gates  = qsolver.find_resources()
        # Compute quantum resources for pattern encoding
        qsolver = ExactCoverQuantumSolver(sudoku, simple=False, pattern=True)
        pat_num_qubits, pat_mcx_gates, pat_total_gates = qsolver.find_resources()
        return sim_num_qubits, sim_mcx_gates, sim_total_gates, pat_num_qubits, pat_mcx_gates, pat_total_gates
    
    def init_ibm(self, ibm_token):
        """
        Returns IBM backends available to your account
        """
        QiskitRuntimeService.save_account(channel="ibm_quantum", token=ibm_token, overwrite=True)
        available_backend_list = IBMQBackend.available_devices()
        return available_backend_list
    
    
    def find_transplied_resources(self, sudoku: Sudoku, backend: str = "ibm_brisbane", optimisation_levels: list =[0,1,2]):
        """
        Compute the quantum resources required for solving a given Sudoku puzzle,
        for a specific backend. 
        # Requires setting up IBM Token with QiskitRuntimeService.save_account(channel="ibm_quantum", token=ibm_token, overwrite=True)
        Parameters:
            sudoku: A Sudoku instance representing the puzzle.
        Returns:
        """
        all_resources = {}
        # Compute quantum resources for simple encoding
        qsolver = ExactCoverQuantumSolver(sudoku=sudoku, simple=True, pattern=False)
        sim_num_qubits, sim_mcx_gates, sim_total_gates  = qsolver.find_resources()
        all_resources["simple"] = (sim_num_qubits, sim_mcx_gates, sim_total_gates)
        qsolver.flatten_reg()
        sim_resources = qsolver.find_transp_resources(ibm_backend=backend, optimisation_levels=optimisation_levels)
        for level in optimisation_levels:
            sim_tr_num_qubits, sim_tr_total_gates, sim_tr_depth = sim_resources[level]
            all_resources[f"sim_tr_oplevel_{level}"] = (sim_tr_num_qubits, sim_tr_total_gates, sim_tr_depth)
        # Compute quantum resources for pattern encoding
        qsolver = ExactCoverQuantumSolver(sudoku=sudoku, simple=False, pattern=True)
        pat_num_qubits, pat_mcx_gates, pat_total_gates = qsolver.find_resources()
        all_resources["pattern"] = ((pat_num_qubits, pat_mcx_gates, pat_total_gates))
        qsolver.flatten_reg()
        pat_resources = qsolver.find_transp_resources(ibm_backend=backend, optimisation_levels=optimisation_levels)
        for level in optimisation_levels:
            patt_tr_num_qubits, patt_tr_total_gates, patt_tr_depth = pat_resources[level]
            all_resources[f"pat_tr_oplevel_{level}"] = (patt_tr_num_qubits, patt_tr_total_gates, patt_tr_depth)
        return all_resources

    def generate_data(self, num_empty_cells: int = None, num_cells_range: tuple = (1, 8)) -> pd.DataFrame:
        """
        Generate Sudoku puzzles and compute quantum resources for each puzzle.
        
        Parameters:
        - num_empty_cells: Specific number of missing cells to generate puzzles with (optional).
        - num_cells_range: Tuple specifying a range of missing cells if num_empty_cells is not provided.
        
        Returns:
        - A DataFrame containing the quantum resource data for the generated puzzles.
        """
        if num_empty_cells is not None:
            if not isinstance(num_empty_cells, int) or num_empty_cells < 0:
                raise ValueError("num_empty_cells must be a non-negative integer.")
        if not (isinstance(num_cells_range, tuple) and len(num_cells_range) == 2):
            raise ValueError("num_cells_range must be a tuple of two integers.")

        if num_empty_cells is None:
            # Loop over a range of missing cells if num_empty_cells is not specified
            for i in range(num_cells_range[0], num_cells_range[1] + 1):
                for _ in tqdm(range(self.num_puzzles), desc=f"Generating puzzles with {i} missing cells"):
                    sudoku = Sudoku(grid_size=self.size, num_missing_cells=i)
                    self._append_quantum_resources(sudoku)
        else:
            # Generate data for a specific number of missing cells
            for _ in tqdm(range(self.num_puzzles), desc=f"Generating puzzles with {num_empty_cells} missing cells"):
                sudoku = Sudoku(grid_size=self.size, num_missing_cells=num_empty_cells)
                self._append_quantum_resources(sudoku)

        return pd.DataFrame(self.data_list)

    def insert_into_sql(self) -> None:
        """
        Insert the data into the 'quantum_resources' table in the database.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            # Ensure the table exists
            self._create_table_if_not_exists()
            # Convert data list to DataFrame
            df = pd.DataFrame(self.data_list)
            # Define data types for the DataFrame columns
            dtype = {
                'missing_cells': Integer(),
                'num_qubits_simple_encoding': Integer(),
                'mcx_gates_simple_encoding': Integer(),
                'total_gates_simple_encoding': Integer(),
                'num_qubits_pattern_encoding': Integer(),
                'mcx_gates_pattern_encoding': Integer(),
                'total_gates_pattern_encoding': Integer(),
            }
            # Insert data into the SQL table
            df.to_sql('quantum_resources', self.engine, if_exists='append', index=False, dtype=dtype, method='multi')
        except Exception as e:
            logger.error(f"Error inserting data into SQL: {e}")
            raise

    def load_db_table_to_df(self, table_name: str) -> pd.DataFrame:
        """
        Load data from a database table into a DataFrame.
        
        Parameters:
        - table_name: The name of the table to load data from.
        
        Returns:
        - A DataFrame containing the data from the specified table.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error loading table {table_name}: {e}")
            raise

    def _append_quantum_resources(self, sudoku: Sudoku) -> None:
        """
        Append quantum resource data for the given Sudoku puzzle to the data list.
        
        Parameters:
        - sudoku: A Sudoku instance representing the puzzle to be solved.
        """
        # Compute quantum resources for the given Sudoku puzzle using both encodings
        sim_num_qubits, sim_mcx_gates, sim_total_gates,  \
        pat_num_qubits, pat_mcx_gates, pat_total_gates  = self.find_quantum_resources(sudoku)
        
        # Append the computed data to the data list
        self.data_list.append({
            'missing_cells': sudoku.num_missing_cells,
            'num_qubits_simple_encoding': sim_num_qubits,
            'mcx_gates_simple_encoding': sim_mcx_gates,
            'total_gates_simple_encoding': sim_total_gates,
            'num_qubits_pattern_encoding': pat_num_qubits,
            'mcx_gates_pattern_encoding': pat_mcx_gates,
            'total_gates_pattern_encoding': pat_total_gates,
        })

    def _create_table_if_not_exists(self) -> None:
        """
        Create the 'quantum_resources' table in the database if it does not already exist.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        # Create all tables defined by the Base class
        Base.metadata.create_all(self.engine)

class QuantumDataAnalysis:
    """
    Class for analyzing and visualizing quantum resource metrics.
    """
    def __init__(self, df: pd.DataFrame):
        # Initialize with a DataFrame containing quantum resource data
        self.df = df

    def get_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics of the DataFrame.
        
        Returns:
        - A DataFrame containing descriptive statistics for each column.
        """
        return self.df.describe()

    def plot_distributions_pdf(self,file_path = None) -> None:
        """
        Plot distributions of key quantum resource metrics and save them all into a single PDF report.
        
        The method generates histograms for specified metrics, a correlation heatmap, and a pairplot.
        It is optimized for handling large datasets by reducing histogram bins, lowering DPI, 
        adjusting figure sizes, and downsampling data for the pairplot.
        
        Requirements:
            - self.df: pandas DataFrame containing the required metrics as columns.
        
        Outputs:
            - 'quantum_resource_distributions.pdf' in the current working directory.
        """
        pdf_filename = 'quantum_resource_distributions.pdf'
        metrics = [
            'num_qubits_simple_encoding', 'mcx_gates_simple_encoding', 'total_gates_simple_encoding',
            'num_qubits_pattern_encoding', 'mcx_gates_pattern_encoding', 'total_gates_pattern_encoding'
        ]
        
        # Reduce the number of bins for histograms and set default plot settings
        bins = 20  # Set to a reasonable number of bins
        dpi = 80   # Lower DPI for saved plots
        figure_size = (8, 6)  # Set a smaller figure size
        
        with PdfPages(pdf_filename) as pdf:
            # Loop through each metric to plot its distribution
            for metric in metrics:
                try:
                    plt.figure(figsize=figure_size)  # Set the figure size
                    # Plot the distribution with fewer bins and disable KDE if not necessary
                    sns.histplot(data=self.df, x=metric, kde=True, bins=bins)
                    plt.title(f'Distribution of {metric}')
                    pdf.savefig(dpi=dpi)  # Save plot to PDF with lower resolution
                    plt.close()  # Close the figure to free memory
                except Exception as e:
                    logger.error(f"Error plotting {metric}: {e}")
                    plt.close()  # Ensure figure is closed even if there's an error

            try:
                # Correlation heatmap
                plt.figure(figsize=figure_size)
                correlation = self.df.corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Heatmap')
                pdf.savefig(dpi=dpi)  # Save heatmap to PDF
                plt.close()  # Close the figure to free memory
            except Exception as e:
                logger.error(f"Error plotting correlation heatmap: {e}")
                plt.close()

            try:
                # Pairplot (this can be intensive, so reduce the sample size if needed)
                small_df = self.df.sample(frac=0.1) if len(self.df) > 1000 else self.df  # Downsample large datasets
                sns.pairplot(small_df)
                pdf.savefig(dpi=dpi)  # Save pairplot to PDF
                plt.close()  # Close the figure to free memory
            except Exception as e:
                logger.error(f"Error plotting pair plots: {e}")
                plt.close()
                
    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute the correlation matrix of the DataFrame.
        
        Returns:
        - A DataFrame representing the correlation coefficients between features.
        """
        return self.df.corr()

    def plot_distributions(self) -> None:
        """
        Plot distributions of key quantum resource metrics and save them as image files.
        """
        metrics = [
            'num_qubits_simple_encoding', 'mcx_gates_simple_encoding', 'total_gates_simple_encoding', 
            'num_qubits_pattern_encoding', 'mcx_gates_pattern_encoding', 'total_gates_pattern_encoding'
        ]
        for metric in metrics:
            # Plot the distribution of each metric
            sns.histplot(data=self.df, x=metric, kde=True)
            plt.title(f'Distribution of {metric}')
            plt.savefig(f'{metric}_distribution.png')
            plt.close()
                
    def pair_plots(self) -> None:
        """
        Generate pair plots to visualize relationships between correlated metrics for 
        simple encoding and pattern encoding, then save the plots separately.
        """
        # Define the metric groups for each encoding scheme
        simple_encoding_metrics = [
            'num_qubits_simple_encoding', 
            'mcx_gates_simple_encoding',
            'total_gates_simple_encoding', 
        ]
        
        pattern_encoding_metrics = [
            'num_qubits_pattern_encoding', 
            'mcx_gates_pattern_encoding',
            'total_gates_pattern_encoding', 

        ]
        
        # Generate and save pair plot for Simple Encoding Metrics
        sns.pairplot(self.df[simple_encoding_metrics])
        plt.title('Simple Encoding Metrics Pair Plot')
        plt.savefig('pair_plots_simple_encoding.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Generate and save pair plot for Pattern Encoding Metrics
        sns.pairplot(self.df[pattern_encoding_metrics])
        plt.title('Pattern Encoding Metrics Pair Plot')
        plt.savefig('pair_plots_pattern_encoding.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def full_correlation_heatmap(self,file_name = 'full_correlation_heatmap.png') -> None:
        """
        Plot a heatmap showing correlations between different features and save it as an image file.
        """
        # Compute the correlation matrix and plot it as a heatmap
        correlation = self.df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        # plt.title('Correlation Heatmap of Quantum Resources')
        plt.show()
        plt.savefig(file_name, dpi=80)
        plt.show()
        plt.close()
        
    def plot_correlation_heatmaps(self) -> None:
        """
        Plot separate heatmaps showing correlations between correlated metrics for 
        simple encoding and pattern encoding, then save the plots as image files.
        """
        # Define the metric groups for each encoding scheme
        simple_encoding_metrics = [
            'num_qubits_simple_encoding', 
            'mcx_gates_simple_encoding',
            'total_gates_simple_encoding'
        ]
        
        pattern_encoding_metrics = [
            'num_qubits_pattern_encoding',
            'mcx_gates_pattern_encoding',
            'total_gates_pattern_encoding' 
        ]
        
        # Plot and save heatmap for Simple Encoding Metrics
        correlation_simple = self.df[simple_encoding_metrics].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_simple, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap - Simple Encoding')
        plt.savefig('correlation_heatmap_simple_encoding.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Plot and save heatmap for Pattern Encoding Metrics
        correlation_pattern = self.df[pattern_encoding_metrics].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_pattern, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap - Pattern Encoding')
        plt.savefig('correlation_heatmap_pattern_encoding.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_interactive_correlation_heatmap(self) -> None:
        """
        Plot an interactive heatmap showing correlations between different features.
        """
        correlation = self.df.corr()
        fig = px.imshow(correlation, 
                        text_auto=True, 
                        title='Interactive Correlation Heatmap of Quantum Resources')
        fig.show()

    def perform_t_tests(self) -> None:
        """
        Perform t-tests to compare the quantum resources between different encodings (simple vs pattern).
        """
        from scipy.stats import ttest_ind
        # Loop through key metrics and perform t-tests between simple and pattern encodings
        for column in ['num_qubits_simple_encoding', 'mcx_gates_simple_encoding', 'total_gates_simple_encoding']:
            pattern_column = column.replace('simple', 'pattern')
            t_stat, p_value = ttest_ind(self.df[column], self.df[pattern_column], equal_var=False, nan_policy='omit')
            logger.info(f"T-test for {column} vs {pattern_column}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# Example use of QuantumDataAnalysis class
# gen_data = GenData(grid_size=2, num_puzzles=10)
# data_df = gen_data.generate_data(num_empty_cells=5)
# analysis = QuantumDataAnalysis(df=data_df)