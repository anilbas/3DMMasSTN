% Wrapper for vl_nnselection block
% inputs{1} :   X     : 1 x 2 x N x b
% obj       :   idx   : 1 x n
% outputs{1}:   y     : 1 x 2 x n x b

classdef selection < dagnn.Layer
    
    properties
        idx = [];
    end
    
    methods
        function outputs = forward(obj, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnselection(gather(inputs{1}), obj.idx) );
            else
                outputs{1} = vl_nnselection(inputs{1}, obj.idx);
            end
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray( vl_nnselection(gather(inputs{1}), obj.idx, gather(derOutputs{1})) );
            else
                derInputs{1} = vl_nnselection(inputs{1}, obj.idx, derOutputs{1});
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes = {1 inputSizes{1}(2) length(obj.idx) inputSizes{1}(4)} ;
        end
        
        function obj = selection(varargin)
            obj.load(varargin);
        end
    end
end
