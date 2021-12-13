import csv

class Action_Space():
    def __init__(self, csv_file):
       self.csv_file = csv_file
       self._range = {}
       with open(csv_file) as f:
           for row in csv.DictReader(f):
               range = int(row['ofm_allocation']) + int(row['weights_allocation'])
               if range == 0:
                   self._range[row['head_names']] = 1
               else:
                   self._range[row['head_names']] = range
               #self._range[row['head_names']] = range
       self.head_names = self._range.keys()
       self._num_heads = len(self.head_names)

    def head_names(self):
	#return list of nodes
        return self.head_names

    def _num_head(self):
        return self._num_heads

    def name2index(self, action, option):
        return 0
