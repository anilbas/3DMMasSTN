% Wrapper for vl_nngrid block
% inputs{1} :   X     : 1 x 3 x N x b
% outputs{1}:   y     : 2 x Ho x Wo x b

classdef resamplegrid < dagnn.Layer
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray(vl_nngrid(gather(inputs{1})));
            else
                outputs{1} = vl_nngrid(inputs{1});
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray( vl_nngrid(gather(inputs{1}), gather(derOutputs{1})) );
            else
                derInputs{1} = vl_nngrid(inputs{1},derOutputs{1});
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = {2 112 112 inputSizes{1}(4)} ; % Note: We may want to set this dynamically as well
        end
        
        function obj = resamplegrid(varargin)
            obj.load(varargin);
        end
    end
end
