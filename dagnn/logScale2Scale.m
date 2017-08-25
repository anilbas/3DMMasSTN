% Wrapper for vl_nnlogScale2Scale block
% inputs{1} :   s     : 1 x 1 x 1 x b
% outputs{1}:   y     : 1 x 1 x 1 x b

classdef logScale2Scale < dagnn.Layer
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnlogScale2Scale(gather(inputs{1})) );
            else
                outputs{1} = vl_nnlogScale2Scale(inputs{1});
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray( vl_nnlogScale2Scale(gather(inputs{1}), gather(derOutputs{1})) );
            else
                derInputs{1} = vl_nnlogScale2Scale(inputs{1}, derOutputs{1});
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = inputSizes{1};
        end
        
        function obj = logScale2Scale(varargin)
            obj.load(varargin);
        end
    end
end
