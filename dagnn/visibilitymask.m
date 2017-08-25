% Wrapper for vl_nnvisibilitymask block
% inputs{1} :   X     : 1 x 3 x n x b (Rotated vertices)
% obj       :   faces : nfaces x 3
% outputs{1}:   y     : 112 x 112 x 3 x b

classdef visibilitymask < dagnn.Layer
    
    properties
        faces = [];
    end
    
    methods
        function outputs = forward(obj, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnvisibilitymask(gather(inputs{1}), obj.faces) );
            else
                outputs{1} = vl_nnvisibilitymask(inputs{1}, obj.faces);
            end
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray( vl_nnvisibilitymask(gather(inputs{1}), obj.faces, gather(derOutputs{1})) );
            else
                derInputs{1} = vl_nnvisibilitymask(inputs{1}, obj.faces, derOutputs{1});
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes = {112 112 3 inputSizes{1}(4)} ; % Note: We may want to set this dynamically as well
        end
        
        function obj = visibilitymask(varargin)
            obj.load(varargin);
        end
    end
end
