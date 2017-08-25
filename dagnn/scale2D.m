% Wrapper for vl_nnscale2D block
% inputs{1} :   X     : 1 x 2 x n x b
% inputs{2} :   S     : 1 x 1 x 1 x b
% outputs{1}:   y     : 1 x 2 x n x b

classdef scale2D < dagnn.Layer
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnscale2D(gather(inputs{1}), gather(inputs{2})) );
            else
                outputs{1} = vl_nnscale2D(inputs{1}, inputs{2});
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                [y,dsdy] =  vl_nnscale2D(gather(inputs{1}), gather(inputs{2}), gather(derOutputs{1}));
                derInputs = {gpuArray(y),gpuArray(dsdy)};
            else
                [y,dsdy] =  vl_nnscale2D(inputs{1}, inputs{2}, derOutputs{1});
                derInputs = {y,dsdy};
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = inputSizes{1};
        end
        
        function obj = scale2D(varargin)
            obj.load(varargin);
        end
    end
end
