% Wrapper for vl_nnvisibility block
% inputs{1} :   X     : 112 x 112 x 3 x b
% inputs{2} :   V     : 112 x 112 x 3 x b
% outputs{1}:   y     : 112 x 112 x 3 x b

classdef visibility < dagnn.Layer
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnvisibility(gather(inputs{1}), gather(inputs{2})) );
            else
                outputs{1} = vl_nnvisibility(inputs{1}, inputs{2});
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                [y,dvdy] = vl_nnvisibility(gather(inputs{1}), gather(inputs{2}), gather(derOutputs{1}));
                derInputs = {gpuArray(y),gpuArray(dvdy)};
            else
                [y,dvdy] = vl_nnvisibility(inputs{1}, inputs{2}, derOutputs{1});
                derInputs = {y,dvdy};
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = inputSizes{1};
        end
        
        function obj = visibility(varargin)
            obj.load(varargin);
        end
    end
end
