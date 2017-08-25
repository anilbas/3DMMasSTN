% Wrapper for vl_nnprojection block
% inputs{1} :   X     : 1 x 3 x n x b
% outputs{1}:   y     : 1 x 2 x n x b

classdef projection < dagnn.Layer
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnprojection(gather(inputs{1})) );
            else
                outputs{1} = vl_nnprojection(inputs{1});
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray( vl_nnprojection(gather(inputs{1}), gather(derOutputs{1})) );
            else
                derInputs{1} =  vl_nnprojection(inputs{1}, derOutputs{1});
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = {1 2 inputSizes{1}(3) inputSizes{1}(4)} ;
        end
        
        function obj = projection(varargin)
            obj.load(varargin);
        end
    end
end
