# -*- coding: utf-8 -*-


class CSummaryNode(object):
    def __init__(self, name=str()):
        self._name = name
        self._input_shape = None
        self._output_shape = None
        self._parameter_quantity = 0
        self._inference_memory = 0
        self._MAdd = 0
        self._duration = 0
        self._duration_percent = 0

        self._children = list()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def input_shape(self):
        if len(self._children) == 0:  # leaf
            return self._input_shape
        else:
            return self._children[0].input_shape
    
    @input_shape.setter
    def input_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        self._input_shape = input_shape
        
    @property
    def output_shape(self):
        if len(self._children) == 0:  # leaf
            return self._output_shape
        else:
            return self._children[-1].output_shape
    
    @output_shape.setter
    def output_shape(self, output_shape):
        assert isinstance(output_shape, (list, tuple))
        self._output_shape = output_shape
        
    @property
    def parameter_quantity(self):
        # return self.parameters_quantity
        total_parameter_quantity = self._parameter_quantity
        for child in self._children:
            total_parameter_quantity += child.parameter_quantity
        return total_parameter_quantity
    
    @parameter_quantity.setter
    def parameter_quantity(self, parameter_quantity):
        assert parameter_quantity >= 0
        self._parameter_quantity = parameter_quantity

    @property
    def inference_memory(self):
        total_inference_memory = self._inference_memory
        for child in self._children:
            total_inference_memory += child.inference_memory
        return total_inference_memory

    @inference_memory.setter
    def inference_memory(self, inference_memory):
        self._inference_memory = inference_memory

    @property
    def MAdd(self):
        total_MAdd = self._MAdd
        for child in self._children:
            total_MAdd += child.MAdd
        return total_MAdd

    @MAdd.setter
    def MAdd(self, MAdd):
        self._MAdd = MAdd

    @property
    def duration(self):
        total_duration = self._duration
        for child in self._children:
            total_duration += child.duration
        return total_duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    def add_child(self, node):
        assert isinstance(node, CSummaryNode)
        self._children.append(node)


def main():
    pass


if __name__ == "__main__":
    main()
