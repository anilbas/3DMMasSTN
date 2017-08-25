% Wrapper for vl_nnmodel block
% inputs{1} :   X     : 1 x 1 x ndims x b
% obj       :   model : 3 x N (shapePC)
% outputs{1}:   y     : 1 x 3 x N x b

classdef model3D < dagnn.Layer
    
    properties
        model = [];
    end
    
    methods
        function outputs = forward(obj, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} = gpuArray( vl_nnmodel(gather(inputs{1}), obj.model) );
            else
                outputs{1} = vl_nnmodel(inputs{1}, obj.model);
            end
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray( vl_nnmodel(gather(inputs{1}), obj.model, gather(derOutputs{1})) );
            else
                derInputs{1} = vl_nnmodel(inputs{1}, obj.model, derOutputs{1});
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes = {1 3 obj.model.nverts inputSizes{1}(4)} ;
        end
        
        function obj = model3D(varargin)
            obj.load(varargin);
        end
    end
end
