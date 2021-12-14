import csv

class Action_Space():
    def __init__(self, csv_file):
       self.csv_file = csv_file
       self._range = {}
       max_ofm_num = 0
       max_weights_num = 0
       self.head_names = []
       self._num_heads = 0
       with open(csv_file) as f:
           for row in csv.DictReader(f):
               self.head_names.append(row['head_names'])
               self._num_heads = self._num_heads + 1
               if max_ofm_num < int(row['ofm_allocation']):
                   max_ofm_num =  int(row['ofm_allocation'])
               if max_weights_num < int(row['weights_allocation']):
                   max_weights_num =  int(row['weights_allocation'])

       self._range['ofm_allocation'] = max_ofm_num
       self._range['weights_allocation'] = max_weights_num


    def head_names(self):
	#return list of nodes
        return self.head_names

    def _num_head(self):
        return self._num_heads

    def name2index(self, action, option):
        return 0
