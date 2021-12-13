import csv

class Action_Space(Object):
    def __init__(self, csv_file):
       self.csv_file = csv_file
       self._range = {}
       with open(csv_file) as f:
           for row in csv.DictReader(f):
               self._range[row['op_name']] = int(row['ofm_allocations']) + int(row['weights_allocations'])
       self.head_names = self._range.keys()

    def head_names(self):
	#return list of nodes
        return self.head_names

    def _num_head(self):
        return len(self.head_names)

    def name2index(self, action, option)
        return 0
