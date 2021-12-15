import csv

class Action_Space():
    def __init__(self, csv_file):
       self.csv_file = csv_file
       self._range = {}
       self.head_names_ = ['ofm_allocation','weights_allocation']
       self._num_heads = 2
       '''
       with open(csv_file) as f:
           for row in csv.DictReader(f):
               self.head_names.append(row['head_names'])
               self._num_heads = self._num_heads + 1
       '''

       self._range['ofm_allocation'] = 2
       self._range['weights_allocation'] = 2



    def head_names(self):
	#return list of nodes
        return self.head_names_

    def name2index(self, action, option):
        return 0
