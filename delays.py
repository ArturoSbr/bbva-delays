# Imports
import re
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Labels for plot
_labels = {0: 'Keep', 1: 'Batch process', 2: 'Outlier'}
_dow = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday',
    'Friday', 'Saturday', 'Sunday'
]


# Class that checks tables stored in an HDFS
class table_verifier:
    """An instance of this class analyzes the update history of a table.

    This class analyzes the update history of a table stored in an HDFS to help
    the user determine the delay between the moment a new partition is appended
    and the date of the information it contains.
    """
    # Init class
    def __init__(
        self, path, partition_field, partition_format,
        samples=90  # partition_frequency='daily'
    ):
        """
        Instantiate a table_verifier object.

        Parameters
        ----------
        path : str
            The HDFS path where the table is stored.
        partition_field : str
            The date field used to partition the table (e.g. 'load_date').
        partition_format : str
            The date format `partition_field` is written in (e.g. '%Y-%m-%d').
        samples : int
            The number of partitions used to analyze the table's update history
            (after being sorted from newest to oldest).
        """

        # Type checks
        msg = 'Expected {} to be a{}, but got {} instead.'
        assert isinstance(path, str), msg.format(
            'path', ' string', type(path).__name__
        )
        assert isinstance(partition_field, str), msg.format(
            'partition_field', ' string', type(partition_field).__name__
        )
        assert isinstance(partition_format, str), msg.format(
            'partition_format', ' string', type(partition_format).__name__
        )
        assert isinstance(samples, int), msg.format(
            'samples', 'n integer', type(samples).__name__
        )
        assert samples > 0, 'samples must be greater than zero.'

        # Assign to self
        self.path = path
        self.partition_field = partition_field
        self.partition_format = partition_format
        self.samples = samples
        self._fitted = False

    # Fit
    def fit(self):
        """Fit a table_verifier instance on the sampled partitions."""

        # Get dataframe
        df = _ls_to_pd(
            self.path,
            self.partition_field,
            self.partition_format,
            self.samples
        )

        # Drop batch processes (1)
        df['drop'] = (
            df['diff_d1']
            .sort_index(ascending=False)
            .rolling(window=5, min_periods=5)
            .min()
            .gt(0)
            .astype(int)
            .sort_index()
        )

        # Drop outliers (2)
        m = df['drop'].eq(0)
        q25, q75 = np.quantile(df.loc[m, 'diff'], [0.25, 0.75])
        ub = q75 + (q75 - q25) * 1.5
        df.loc[m & df['diff'].ge(ub), 'drop'] = 2

        # Assign as attributes
        self._fitted = True
        self.df = df
        self.delay_avg = df.loc[df['drop'].eq(0), 'diff'].mean()
        self.delay_median = df.loc[df['drop'].eq(0), 'diff'].median()

    # Bar plot of information delay (main plot)
    def plot(self):
        """Bar plot of partitions (x-axis) and information delay (y-axis).

        This method must be called on a fitted table_verifier instance. It will
        display a bar plot of the delay (in days) between partitions' write
        dates and the information they represent.
        """
        if self._fitted:
            for i in range(3):
                mask = self.df['drop'].eq(i)
                plt.bar(
                    self.df.loc[mask, 'date_part'],
                    self.df.loc[mask, 'diff'],
                    label=_labels[i],
                    color=f'C{i}',
                    alpha=1 - (0.8 * int(i > 0))
                )

            # Aesthetics
            plt.xlabel(self.partition_field.replace('_', ' ').title())
            plt.ylabel('Information delay (days)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.show()
        else:
            raise ValueError('The instance has not been fitted yet.')

    # Histogram
    def hist(self, show_outliers=False):
        """Histogram of the table's delay (measeured in days).

        This method must be called on a fitted table_verifier instance. It will
        display a histogram of the delay (in days) between partitions' write
        dates and the information they represent.

        Parameters
        ----------
        show_outliers : bool
            Hide or show the outliers spotted during the fit process. Outliers
            are hidden by default.
        """
        if self._fitted:
            mask = (
                self.df['drop'].ge(0) if show_outliers
                else self.df['drop'].eq(0)
            )
            plt.hist(self.df.loc[mask, 'diff'], bins=30)
            plt.axvline(self.delay_avg, ls='--', color='C1', label='Average')
            plt.axvline(self.delay_median, ls='--', color='C2', label='Median')
            plt.xlabel('Information delay (days)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
        else:
            raise ValueError('The instance has not been fitted yet.')

    # Frequency per day of week
    def dow(self, show_outliers=False):
        if self._fitted:
            mask = (
                self.df['drop'].ge(0) if show_outliers
                else self.df['drop'].eq(0)
            )
            x = self.df[mask].groupby('dow').size()
            plt.bar(x.index, x.values)
            plt.xticks(ticks=x.index, labels=_dow, rotation=45)
            plt.xlabel('Day of week when partition was written')
            plt.ylabel('Frequency')
            plt.show()
        else:
            raise ValueError('The instance has not been fitted yet.')


# Verify path
def ls(path):
    """List the partitions available in the path provided by the user."""
    # Run -ls
    cmd = subprocess.run(
        ['hdfs', 'dfs', '-ls', path],
        capture_output=True
    )

    # Return output
    if cmd.returncode == 0:
        output = str(cmd.stdout)
        return output
    else:
        raise FileNotFoundError(cmd.sterror)


# Parse ls output
def _parse(console_output, partition_field):
    """Parse the output returned by `ls`."""
    # Check type
    msg = 'Expected console_output to be a string, but got {} instead.'
    assert isinstance(
        console_output, str
    ), msg.format(type(console_output).__name__)

    # Split output and keep partition lines
    files = console_output.split('\\n')
    files = [
        re.split(r'\s+', line.strip())
        for line in files if partition_field in line
    ]

    # Check if partition field exists
    if len(files) == 0:
        msg = f'There are no files partitioned by "{partition_field}".'
        raise ValueError(msg)
    else:
        return files


# Transform parsed text to DataFrame
def _ls_to_pd(path, partition_field, partition_format, samples):

    # Send process and transform to pandas
    console_output = ls(path)
    parsed_list = _parse(console_output, partition_field)

    # Check type
    msg = 'Expected parsed_list to be a list, but got {} instead.'
    assert isinstance(
        parsed_list, list
    ), msg.format(type(parsed_list).__name__)

    # Create DataFrame
    df = pd.DataFrame(
        data=parsed_list[-samples:],
        columns=[
            'permissions', 'replication', 'owner', 'group',
            'size', 'mod_date', 'mod_hour', 'path'
        ]
    )

    # Assign columns
    try:
        df = df.assign(
            date_mod=pd.to_datetime(
                arg=df['mod_date'] + df['mod_hour'],
                format='%Y-%m-%d%H:%M'
            ),  # TO DO: Convert to timezone
            date_part=pd.to_datetime(
                arg=df['path'].str.split(
                    partition_field + '=', expand=True
                )[1],
                format=partition_format
            )
        )
        df = df.assign(
            dow=df['date_mod'].dt.dayofweek,
            diff=(
                (df['date_mod'] - df['date_part']).dt.total_seconds()
                / (24 * 60 * 60)
            )
        )
        df['diff_d1'] = df['diff'].shift(1) - df['diff']

    except ValueError:
        msg = 'partition_format {} does not match console output.'
        raise ValueError(msg.format(partition_format))

    # Return dataframe
    return df
