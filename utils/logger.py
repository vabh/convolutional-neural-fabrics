import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
class Logger:
    def __init__(self, log_file, names=None, delimiter='\t'):
        assert log_file is not None
        if names is None:
            names = []

        self.log_file = log_file
        self.names = names
        self.delim = delimiter
        self.fields = len(names)

        header = self._gather_values(self.names, prefix='#')
        with open(log_file, 'w') as f:
            f.write(header + '\n')

    def _gather_values(self, vals, prefix=''):
        output = ''
        for value in vals:
            output =  output + self.delim + str(value)
        output = prefix + output
        return output

    def add(self, vals):
        assert len(vals) == self.fields
        output = self._gather_values(vals)
        with open(self.log_file, 'a') as f:
            f.write(output + '\n')

    def plot(self):
        data = np.loadtxt(self.log_file, skiprows=1)
        plt.clf()
        p = plt.plot(data)
        plt.legend(p, self.names)
        plt.grid()
        plt.savefig(self.log_file+'.png', format='png')

if __name__ == '__main__':
    l = Logger('test.log', names=['a', 'b', 'c'])
    for i in range(4):
        l.add(['a', 'b', 'c'])


