class QuantumDataAnalysis:
    """
    Class for analyzing and visualizing quantum resource metrics from Sudoku solver data.
    
    This class provides methods to:
    - Generate statistical summaries
    - Create visualizations (distributions, correlations)
    - Export analysis results to PDF
    """
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        required_columns = ['solver', 'resource_type', 'puzzle_hash']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        self.df = df.copy()  # Defensive copy

    def plot_analysis_pdf(self, output_file: str = 'quantum_resource_analysis.pdf'):
        """
        Generate a comprehensive PDF report with all relevant visualizations.
        
        Args:
            output_file (str): Path where to save the PDF report.
        """
        with PdfPages(output_file) as pdf:
            # Resource distributions by solver
            for solver in self.df['solver'].unique():
                solver_data = self.df[self.df['solver'] == solver]
                
                # Theoretical resources
                theoretical = solver_data[solver_data['resource_type'] == 'theoretical']
                if not theoretical.empty and theoretical['n_qubits'].notna().any():
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(f'{solver} - Theoretical Resources')
                    
                    # Only plot if we have valid data
                    if theoretical['n_qubits'].notna().any():
                        sns.histplot(data=theoretical.dropna(subset=['n_qubits']), x='n_qubits', ax=axes[0])
                    axes[0].set_title('Number of Qubits')
                    
                    if theoretical['n_gates'].notna().any():
                        sns.histplot(data=theoretical.dropna(subset=['n_gates']), x='n_gates', ax=axes[1])
                    axes[1].set_title('Number of Gates')
                    
                    if theoretical['MCX_gates'].notna().any():
                        sns.histplot(data=theoretical.dropna(subset=['MCX_gates']), x='MCX_gates', ax=axes[2])
                    axes[2].set_title('MCX Gates')
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                else:
                    # Create placeholder plot for no data
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.text(0.5, 0.5, f'No theoretical data for {solver}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=16)
                    plt.title(f'{solver} - No Theoretical Data Available')
                    pdf.savefig()
                    plt.close()

                # Transpiled resources by optimization level
                transpiled = solver_data[solver_data['resource_type'] == 'transpiled']
                if not transpiled.empty:
                    for level in sorted(transpiled['optimisation_level'].unique()):
                        if pd.isna(level):
                            continue
                        level_data = transpiled[transpiled['optimisation_level'] == level]
                        
                        if level_data.empty:
                            continue
                            
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        fig.suptitle(f'{solver} - Transpiled Resources (Level {level})')
                        
                        # Only plot if we have valid data
                        if level_data['n_qubits'].notna().any():
                            sns.histplot(data=level_data.dropna(subset=['n_qubits']), x='n_qubits', ax=axes[0])
                        axes[0].set_title('Number of Qubits')
                        
                        if level_data['n_gates'].notna().any():
                            sns.histplot(data=level_data.dropna(subset=['n_gates']), x='n_gates', ax=axes[1])
                        axes[1].set_title('Number of Gates')
                        
                        if level_data['depth'].notna().any():
                            sns.histplot(data=level_data.dropna(subset=['depth']), x='depth', ax=axes[2])
                        axes[2].set_title('Circuit Depth')
                        
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

            # Correlation heatmaps
            for resource_type in ['theoretical', 'transpiled']:
                data = self.df[self.df['resource_type'] == resource_type]
                if not data.empty:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:  # Only create heatmap if we have multiple numeric columns
                        corr_data = data[numeric_cols].corr()
                        if not corr_data.empty and corr_data.shape[0] > 1:
                            plt.figure(figsize=(10, 8))
                            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
                            plt.title(f'Correlation Heatmap - {resource_type.capitalize()} Resources')
                            plt.tight_layout()
                            pdf.savefig()
                            plt.close()

    def get_error_summary(self) -> pd.DataFrame:
        """
        Generate summary of errors encountered during processing.
        
        Returns:
            pd.DataFrame: Summary of errors by solver and error type.
        """
        error_data = self.df[self.df['error'].notna()]
        if error_data.empty:
            return pd.DataFrame()
        
        return error_data.groupby(['solver', 'resource_type', 'error']).size().reset_index(name='count')

    def get_resource_efficiency_comparison(self) -> pd.DataFrame:
        """
        Compare resource efficiency across solvers.
        
        Returns:
            pd.DataFrame: Resource efficiency metrics by solver.
        """
        theoretical = self.df[
            (self.df['resource_type'] == 'theoretical') & 
            (self.df['error'].isna())
        ]
        
        if theoretical.empty:
            return pd.DataFrame()
        
        efficiency_metrics = theoretical.groupby('solver').agg({
            'n_qubits': ['mean', 'std', 'min', 'max'],
            'n_gates': ['mean', 'std', 'min', 'max'],
            'MCX_gates': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        return efficiency_metrics

    def get_optimization_impact(self) -> pd.DataFrame:
        """
        Analyze the impact of different optimization levels on resource requirements.
        
        Returns:
            pd.DataFrame: Resource requirements by solver and optimization level.
        """
        transpiled = self.df[
            (self.df['resource_type'] == 'transpiled') & 
            (self.df['error'].isna()) &
            (self.df['optimisation_level'].notna())
        ]
        
        if transpiled.empty:
            return pd.DataFrame()
        
        optimization_impact = transpiled.groupby(['solver', 'optimisation_level']).agg({
            'n_qubits': ['mean', 'std'],
            'n_gates': ['mean', 'std'],
            'depth': ['mean', 'std']
        }).round(2)
        
        return optimization_impact

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Generate summary statistics for all numeric columns, grouped by solver and resource type.
        
        Returns:
            pd.DataFrame: Summary statistics including count, mean, std, min, max, etc.
        """
        return self.df.groupby(['solver', 'resource_type']).describe()

    def export_analysis(self, output_dir: str = '.') -> None:
        """
        Export all analyses to files in the specified directory.
        
        Args:
            output_dir (str): Directory where to save the analysis files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Export PDF report
            self.plot_analysis_pdf(os.path.join(output_dir, 'quantum_resource_analysis.pdf'))
            
            # Export summary statistics
            stats = self.get_summary_stats()
            stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
            
            # Export error summary
            error_summary = self.get_error_summary()
            if not error_summary.empty:
                error_summary.to_csv(os.path.join(output_dir, 'error_summary.csv'), index=False)
            
            # Export efficiency comparison
            efficiency = self.get_resource_efficiency_comparison()
            if not efficiency.empty:
                efficiency.to_csv(os.path.join(output_dir, 'resource_efficiency.csv'))
            
            # Export optimization impact
            opt_impact = self.get_optimization_impact()
            if not opt_impact.empty:
                opt_impact.to_csv(os.path.join(output_dir, 'optimization_impact.csv'))
            
            logger.info(f"Analysis exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export analysis to {output_dir}: {e}")
            raise


import json
import pandas as pd
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from sudoku_nisq import Sudoku

# Constants
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3
SUPPORTED_FORMATS = ["csv"]

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    solvers: list[str]
    optimisation_levels: Optional[list[int]] = None
    include_transpiled: bool = True
    save_format: str = "csv"
    filename: str = "sudoku_profiles.csv"
    verbose: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = MAX_RETRIES
    
    def __post_init__(self):
        if self.optimisation_levels is None:
            self.optimisation_levels = [0, 1, 2, 3]
        
        if self.save_format.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.save_format}. Supported: {SUPPORTED_FORMATS}")

class SudokuResourceDatasetBuilder:
    def __init__(self,
                 solvers: list[str],
                 optimisation_levels: list[int] | None = None,
                 include_transpiled: bool = True,
                 save_to: str = "csv",
                 filename: str = "sudoku_profiles.csv",
                 verbose: bool = False) -> None:
        if not solvers:
            raise ValueError("solvers list cannot be empty")
        if not filename:
            raise ValueError("filename cannot be empty")
        
        self.solvers = solvers
        self.optimisation_levels = optimisation_levels if optimisation_levels is not None else [0, 1, 2, 3]
        self.include_transpiled = include_transpiled
        self.save_to = save_to
        self.filename = filename
        self.verbose = verbose

    def _load_existing_data(self) -> set[str]:
        """Load existing puzzle hashes from file."""
        existing_hashes = set()
        if not os.path.exists(self.filename):
            return existing_hashes
            
        try:
            df_existing = pd.read_csv(self.filename)
            existing_hashes = set(df_existing["puzzle_hash"].dropna().unique())
            logger.info(f"Loaded {len(existing_hashes)} existing puzzle hashes from {self.filename}")
        except (pd.errors.EmptyDataError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Failed to load existing file: {e}")
            if self.verbose:
                print("Warning: Failed to load existing file. It will be ignored.")
        except Exception as e:
            logger.error(f"Unexpected error loading existing file: {e}")
            if self.verbose:
                print(f"Unexpected error loading existing file: {e}")
        
        return existing_hashes

    def _save_batch(self, records: list[dict]) -> None:
        """Save a batch of records to the file."""
        if not records:
            return
            
        df_batch = pd.DataFrame(records)
        
        if os.path.exists(self.filename):
            df_existing = pd.read_csv(self.filename)
            df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
        else:
            df_combined = df_batch
            
        df_combined.to_csv(self.filename, index=False)
        logger.info(f"Saved batch of {len(records)} records to {self.filename}")

    def generate_and_profile(self, num_puzzles: int, num_missing_cells: int) -> pd.DataFrame:
        """Generate and analyze Sudoku puzzles using various quantum solvers.
        
        This function generates unique Sudoku puzzles and profiles them using different
        quantum solving approaches, collecting both theoretical and transpiled resource
        requirements for each solver.
        
        Args:
            num_puzzles (int): Number of unique puzzles to generate.
            num_missing_cells (int): Number of cells to leave empty in each puzzle.
        
        Returns:
            pd.DataFrame: DataFrame containing profiling results for all puzzles
                and solvers, including resource requirements and any errors encountered.
        
        Raises:
            ValueError: If an unsupported save_to format is specified.
        """
        if num_puzzles <= 0:
            raise ValueError("num_puzzles must be greater than 0")
        if num_missing_cells <= 0 or num_missing_cells >= 81:
            raise ValueError("num_missing_cells must be between 1 and 80")
            
        # Load existing data and collect known puzzle hashes
        existing_hashes = self._load_existing_data()

        records: list[dict] = []
        generated_count = 0
        
        # Use tqdm for progress tracking
        pbar = tqdm(total=num_puzzles, desc="Generating puzzles")

        while generated_count < num_puzzles:
            # Process in batches to manage memory for large datasets
            if len(records) >= DEFAULT_BATCH_SIZE:
                self._save_batch(records)
                records = []
            try:
                # Generate a new puzzle with specified number of empty cells
                s = Sudoku(num_missing_cells=num_missing_cells, canonicalize=True)
                puzzle_can_str = json.dumps(s.puzzle.board)
                puzzle_hash = s.get_hash()

                # Skip duplicates against past runs and within this run
                if puzzle_hash in existing_hashes:
                    if self.verbose:
                        logger.info(f"Skipping duplicate puzzle with hash {puzzle_hash}")
                    continue

                # Mark this hash so we won't generate it again in this loop
                existing_hashes.add(puzzle_hash)
                logger.debug(f"Generated new puzzle with hash {puzzle_hash}")

            except (ValueError, RuntimeError) as e:
                logger.error(f"Puzzle generation failed: {e}")
                if self.verbose:
                    print(f"Puzzle generation failed: {e}")
                # Record failure for each solver
                for key in self.solvers:
                    records.append({
                        "puzzle_hash": None,
                        "solver": key,
                        "resource_type": "theoretical",
                        "n_qubits": None,
                        "MCX_gates": None,
                        "n_gates": None,
                        "depth": None,
                        "optimisation_level": None,
                        "backend": None,
                        "error": f"Puzzle generation failed: {str(e)}",
                        "canonical_puzzle": None
                    })
                continue
            except Exception as e:
                logger.critical(f"Unexpected error in puzzle generation: {e}")
                raise  # Re-raise unexpected errors

            # Process each solver for the current puzzle
            for key in self.solvers:
                try:
                    # Initialize the solver and get its main circuit
                    getattr(s, f"init_{key}")()
                    print(f"init_{key}")  # Debug print to confirm initialization
                    solver = getattr(s, key)
                    
                    solver.main_circuit = solver.get_circuit()
                    
                except Exception as e:
                    logger.error(f"Solver {key} initialization failed: {e}")
                    # Record solver initialization failure
                    records.append({
                        "puzzle_hash": puzzle_hash,
                        "solver": key,
                        "resource_type": "theoretical",
                        "n_qubits": None,
                        "MCX_gates": None,
                        "n_gates": None,
                        "depth": None,
                        "optimisation_level": None,
                        "backend": None,
                        "error": f"Solver init failed: {str(e)}",
                        "canonical_puzzle": puzzle_can_str,
                    })
                    continue

                # Calculate and record theoretical resource requirements
                try:
                    tr = solver.resource_estimation()
                except Exception as e:
                    logger.error(f"Resource calculation failed for {key}: {e}")
                    tr = {"n_qubits": None, "MCX_gates": None, "n_gates": None, "error": str(e)}

                records.append({
                    "puzzle_hash": puzzle_hash,
                    "solver": key,
                    "resource_type": "theoretical",
                    "n_qubits": tr.get("n_qubits"),
                    "MCX_gates": tr.get("MCX_gates"),
                    "n_gates": tr.get("n_gates"),
                    "depth": None,
                    "optimisation_level": None,
                    "backend": None,
                    "error": tr.get("error", None),
                    "canonical_puzzle": puzzle_can_str,
                })

                # Calculate and record transpiled resource requirements if requested
                if self.include_transpiled:
                    try:
                        tx_list = solver.find_transpiled_resources(optimisation_levels=self.optimisation_levels)
                    except Exception as e:
                        logger.error(f"Transpilation failed for {key}: {e}")
                        # Create error entries for each optimization level
                        tx_list = [
                            {"optimisation_level": lvl, "backend": getattr(solver, "current_backend", None),
                             "n_qubits": None, "n_gates": None, "depth": None, "error": str(e)}
                            for lvl in self.optimisation_levels
                        ]

                    for meta in tx_list:
                        records.append({
                            "puzzle_hash": puzzle_hash,
                            "solver": key,
                            "resource_type": "transpiled",
                            "n_qubits": meta.get("n_qubits"),
                            "MCX_gates": None,
                            "n_gates": meta.get("n_gates"),
                            "depth": meta.get("depth"),
                            "optimisation_level": meta.get("optimisation_level"),
                            "backend": meta.get("backend"),
                            "error": meta.get("error"),
                            "canonical_puzzle": puzzle_can_str,
                        })

            generated_count += 1
            pbar.update(1)
            logger.info(f"Completed processing puzzle {generated_count}/{num_puzzles}")

        pbar.close()

        # Save any remaining records
        if records:
            self._save_batch(records)

        # Create DataFrame from all saved records
        if os.path.exists(self.filename):
            df = pd.read_csv(self.filename)
        else:
            df = pd.DataFrame()

        # Save final results in the specified format
        if self.save_to.lower() == "csv":
            logger.info(f"Saved results to {self.filename}")
            if self.verbose:
                print(f"Saved to {self.filename}")
        else:
            raise ValueError(f"Unsupported save_to: {self.save_to}")

        return df

