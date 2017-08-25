% Wrapper for vl_nnrotate3D block
% inputs{1} :   X     : 1 x 3 x n x b
% inputs{2} :   R     : 1 x 3 x 3 x b
% outputs{1}:   y     : 1 x 3 x n x b

classdef rotate3D < dagnn.Layer
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnrotate3D(gather(inputs{1}), gather(inputs{2})) );
            else
                outputs{1} = vl_nnrotate3D(inputs{1}, inputs{2});
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                [y,dRdy] = vl_nnrotate3D(gather(inputs{1}), gather(inputs{2}), gather(derOutputs{1}));
                derInputs = {gpuArray(y),gpuArray(dRdy)};
            else
                [y,dRdy] = vl_nnrotate3D(inputs{1}, inputs{2}, derOutputs{1});
                derInputs = {y,dRdy};
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = inputSizes{1};
        end
        
        function obj = rotate3D(varargin)
            obj.load(varargin);
        end
    end
end
