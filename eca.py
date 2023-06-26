import numpy as np

_3bit_input_patterns = [
    (1,1,1),
    (1,1,0),
    (1,0,1),
    (1,0,0),
    (0,1,1),
    (0,1,0),
    (0,0,1),
    (0,0,0)
]

valid_paddings = [
    '0',
    '1',
    'wrap'
]


class ECA:
    def __init__(self, rule):
        '''
        class for handeling a ECA rule
        rule can either be an int or a binary list
        rule must be a valid ECA rule
        '''

        if isinstance(rule, list):
            if len(rule) == len(_3bit_input_patterns):
                if self.is_binary_list(rule):
                    self.rule = dict(zip(_3bit_input_patterns, rule))
                    self.rule_array = rule
                else:
                    raise Exception('rule is not a binary list')
            else:
                raise Exception(f'{len(rule)} is not a valid length for an eca rule, must be 8')
        elif isinstance(rule, int):
            if rule >= 0 and rule < 256:
                # convert int to bit list
                self.rule_array = self.to_list(rule)
                self.rule = dict(zip(_3bit_input_patterns, self.rule_array))
            else:
                raise Exception(f'ECA int must be between 0 and 255')
        else:
            raise Exception(f'rule must be a list or an int')

    def iterate(self, input, padding='wrap'):
        ''' 
        Applies the rule to the input, input must be a binary list
        wallid paddings for edges are 0, 1 or wrap
        return a binary list with the rule applied
        '''
        if padding not in valid_paddings:
            raise Exception(f'padding must be either: {valid_paddings}')
        
        if not self.is_binary_list(input):
            raise Exception(f'Can not apply the rule to a non binary list')
        
        if (padding == 'wrap'):
            input = np.pad(input, (1, 1), 'constant', constant_values=(input[-1], input[0]))
        elif (padding == '0'):
            input = np.pad(input, (1, 1), 'constant', constant_values=(0, 0))
        elif (padding == '1'):
            input = np.pad(input, (1, 1), 'constant', constant_values=(1, 1))
        
        output = np.zeros_like(input)
        
        for i in range(1, input.shape[0] - 1):
            output[i] = self.rule[tuple(input[i-1 : i+1+1])]

        return list(output[1:-1])

    def is_binary_list(self, ary):
        a = np.array(ary)
        if ((a==0) | (a==1)).all():
            return True
        return False

    def to_list(self, int_rule):
        l = [int(i) for i in list('{0:0b}'.format(int_rule))]
        while len(l) != 8:
            l.insert(0, 0)
        return l

    def __iter__(self):
        return iter(self.rule_array)

    def __int__(self):
        return int("".join(str(x) for x in self.rule_array), 2)

    def __str__(self):
        return self.rule.__str__()



if __name__ == "__main__":

    rule = ECA(82)
    print('class:', rule)
    print('str:  ', str(rule))
    print('int:  ', int(rule))
    print('list: ', list(rule))

    n = [0,0,0,1,0,1,0,1,1,0,1,1,1]
    m = rule.iterate(n)
    print(n)
    print(m)