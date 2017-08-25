% Wrapper for vl_nnr2R block
% inputs{1} :   r     : 1 x 1 x 3 x b
% outputs{1}:   R     : 1 x 3 x 3 x b

classdef r2R < dagnn.Layer
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnr2R(gather(inputs{1})) );
            else
                outputs{1} =  vl_nnr2R(inputs{1});
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray( vl_nnr2R(gather(inputs{1}), gather(derOutputs{1})) );
            else
                derInputs{1} = vl_nnr2R(inputs{1}, derOutputs{1});
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = {1 3 inputSizes{1}(3) inputSizes{1}(4)} ;
        end
        
        function obj = r2R(varargin)
            obj.load(varargin);
        end
    end
end
