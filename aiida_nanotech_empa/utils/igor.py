"""Classes for use with IGOR Pro

Code is based on asetk module by Leopold Talirz
(https://github.com/ltalirz/asetk/blob/master/asetk/format/igor.py)

-Kristjan Eimre
"""

import re

import numpy as np


class FileNotBeginsWithIgorError(OSError):
    def __init__(self, fname):
        super().__init__(f"File {fname} does not begin with 'IGOR'.")


class MissingBeginError(OSError):
    def __init__(self, fname):
        super().__init__(f"Missing 'BEGIN' statement in {fname}.")


def read_wave(lines, fname=None):
    line = lines.pop(0)
    while not re.match("WAVES", line):
        if len(lines) == 0:
            return None
        line = lines.pop(0)
    # 1d or 2d?
    d2 = False
    if "N=" in line:
        d2 = True
        match = re.search(r"WAVES/N=\(([\d, ]+)\)", line)
        grid = match.group(1).split(",")
        grid = np.array(grid, dtype=int)
        name = line.split(")")[-1].strip()
    else:
        name = line.split()[-1]

    line = lines.pop(0).strip()
    if not line == "BEGIN":
        raise MissingBeginError(fname)

    # read data
    datastring = ""
    line = lines.pop(0)
    while not re.match("END", line):
        if len(lines) == 0:
            return None
        if line.startswith("X"):
            return None
        datastring += line
        line = lines.pop(0)
    data = np.array(datastring.split(), dtype=float)
    if d2:
        data = data.reshape(grid)

    # read axes
    axes = []
    line = lines.pop(0)
    matches = re.findall("SetScale.+?(?:;|$)", line)
    for match in matches:
        ax = Axis(None, None, None, None)
        ax.read(match)
        axes.append(ax)

    if d2:
        # read also the second axis
        # is this necessary? can there be 2 lines with "SetScale" ?
        line = lines.pop(0)
        matches = re.findall("SetScale.+?(?:;|$)", line)
        for match in matches:
            ax = Axis(None, None, None, None)
            ax.read(match)
            axes.append(ax)
        return Wave2d(data, axes, name)

    return Wave1d(data, axes, name)


def igor_wave_factory(fname):
    """
    Returns either wave1d or wave2d, corresponding to the input file
    """
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()

    line = lines.pop(0).strip()
    if not line == "IGOR":
        raise FileNotBeginsWithIgorError(fname)

    waves = []
    while len(lines) != 0:
        try:
            wave = read_wave(lines, fname)
            if wave is not None:
                waves.append(wave)
        except Exception as e:
            print("  Error: %.80s..." % str(e))

    return waves


class Axis:
    """Represents an axis of an IGOR wave"""

    def __init__(self, symbol, min_, delta, unit, wavename=None):
        self.symbol = symbol
        self.min = min_
        self.delta = delta
        self.unit = unit
        self.wavename = wavename

    def __str__(self):
        """Prints axis in itx format
        Note: SetScale/P expects minimum value and step-size
        """
        delta = 0 if self.delta is None else self.delta
        s = 'X SetScale/P {symb} {min},{delta}, "{unit}", {name};\n'.format(
            symb=self.symbol,
            min=self.min,
            delta=delta,
            unit=self.unit,
            name=self.wavename,
        )
        return s

    def read(self, string):
        """Read axis from string
        Format:
        X SetScale/P x 0,2.01342281879195e-11,"m", data_00381_Up;
        SetScale d 0,0,"V", data_00381_Up
        """
        match = re.search(
            r"SetScale/?P? (.) ([+-\.\de]+),([+-\.\de]+),[ ]*\"(.*)\",\s*(\w+)", string
        )
        self.symbol = match.group(1)
        self.min = float(match.group(2))
        self.delta = float(match.group(3))
        self.unit = match.group(4)
        self.wavename = match.group(5)


class Wave:
    """A class template for IGOR waves of generic dimension"""

    def __init__(self, data, axes, name=None):
        """Initialize IGOR wave of generic dimension"""
        self.data = data
        self.name = "PYTHON_IMPORT" if name is None else name
        self.axes = axes

    def __str__(self):
        """Print IGOR wave"""
        s = ""
        s += "IGOR\n"

        dimstring = "("
        for sh in self.data.shape:
            dimstring += f"{sh}, "
        dimstring = dimstring[:-2] + ")"

        s += f"WAVES/N={dimstring}  {self.name}\n"
        s += "BEGIN\n"
        s += self.print_data()
        s += "END\n"
        for ax in self.axes:
            s += str(ax)
        return s

    @classmethod
    def read(cls, fname):
        """Read IGOR wave"""
        with open(fname, encoding="utf-8") as f:
            lines = f.readlines()

        line = lines.pop(0).strip()
        if not line == "IGOR":
            raise FileNotBeginsWithIgorError(fname)

        line = lines.pop(0)
        while not re.match("WAVES", line):
            line = lines.pop(0)
        # 1d or 2d?
        waves_str, name = line.split()
        d2 = False
        if "N=" in waves_str:
            d2 = True
            match = re.search(r"WAVES/N=\(([\d,]+)\)", waves_str)
            grid = match.group(1).split(",")
            grid = np.array(grid, dtype=int)

        line = lines.pop(0).strip()
        if not line == "BEGIN":
            raise MissingBeginError(fname)

        # read data
        datastring = ""
        line = lines.pop(0)
        while not re.match("END", line):
            datastring += line
            line = lines.pop(0)
        data = np.array(datastring.split(), dtype=float)
        if d2:
            data = data.reshape(grid)

        # read axes
        line = lines.pop(0)
        matches = re.findall("SetScale.+?(?:;|$)", line)
        axes = []
        for match in matches:
            ax = Axis(None, None, None, None)
            ax.read(match)
            axes.append(ax)

        # the rest is discarded...

        return cls(data, axes, name)

    @property
    def extent(self):
        """Returns extent for plotting"""
        grid = self.data.shape
        extent = []
        for i, g in enumerate(grid):
            ax = self.axes[i]
            extent.append(ax.min)
            extent.append(ax.min + ax.delta * g)

        return np.array(extent)

    def print_data(self):
        """Determines how to print the data block.

        To be implemented by subclasses."""

    def write(self, fname):
        with open(fname, "w", encoding="utf-8") as f:
            f.write(str(self))

    def csv_header(self):
        header = ""
        shape = self.data.shape
        for i_ax, s in enumerate(shape):
            ax = self.axes[i_ax]
            if header != "":
                header += "\n"
            header += "axis %d: %s [unit: %s] [%.6e, %.6e], delta=%.6e, n=%d" % (
                i_ax,
                ax.symbol,
                ax.unit,
                ax.min,
                ax.min + ax.delta * (s - 1),
                ax.delta,
                s,
            )
        return header

    def write_csv(self, fname, fmt="%.6e"):
        np.savetxt(fname, self.data, delimiter=",", header=self.csv_header(), fmt=fmt)


class Wave1d(Wave):
    """1d Igor wave"""

    default_parameters = {
        "xmin": 0.0,
        "xdelta": None,
        "xlabel": "x",
        "ylabel": "y",
    }

    def __init__(self, data=None, axes=None, name="1d", **kwargs):
        """Initialize 1d IGOR wave"""
        super().__init__(data, axes, name)

        self.parameters = self.default_parameters
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise KeyError(f"Unknown parameter {key}")

        if axes is None:
            p = self.parameters
            x = Axis(
                symbol="x",
                min_=p["xmin"],
                delta=p["xdelta"],
                unit=p["xlabel"],
                wavename=self.name,
            )
            self.axes = [x]

    def print_data(self):
        s = ""
        for line in self.data:
            s += f"{float(line):12.6e}\n"

        return s


class Wave2d(Wave):
    """2d Igor wave"""

    default_parameters = {
        "xmin": 0.0,
        "xdelta": None,
        "xlabel": "x",
        "xmax": None,
        "ymin": 0.0,
        "ydelta": None,
        "ylabel": "y",
        "ymax": None,
    }

    def __init__(self, data=None, axes=None, name=None, **kwargs):
        """Initialize 2d Igor wave
        Parameters
        ----------

         * data
         * name
         * xmin, xdelta, xlabel
         * ymin, ydelta, ylabel
        """
        super().__init__(data, axes=axes, name=name)

        self.parameters = self.default_parameters
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise KeyError(f"Unknown parameter {key}")

        if axes is None:
            p = self.parameters

            nx, ny = self.data.shape
            if p["xmax"] is None:
                p["xmax"] = p["xdelta"] * nx
            elif p["xdelta"] is None:
                p["xdelta"] = p["xmax"] / nx

            if p["ymax"] is None:
                p["ymax"] = p["ydelta"] * ny
            elif p["ydelta"] is None:
                p["ydelta"] = p["ymax"] / ny

            x = Axis(
                symbol="x",
                min_=p["xmin"],
                delta=p["xdelta"],
                unit=p["xlabel"],
                wavename=self.name,
            )
            y = Axis(
                symbol="y",
                min_=p["ymin"],
                delta=p["ydelta"],
                unit=p["ylabel"],
                wavename=self.name,
            )
            self.axes = [x, y]

    def print_data(self):
        """Determines how to print the data block"""
        s = ""
        for line in self.data:
            for x in line:
                s += f"{x:12.6e} "
            s += "\n"

        return s
