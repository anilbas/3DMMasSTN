% Wrapper for vl_nnsiamese block

classdef siamese < dagnn.Loss
    
    methods
        function outputs = forward(obj, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} =  gpuArray(vl_nnsiamese(gather(inputs{1})));
            else
                outputs{1} =  vl_nnsiamese(inputs{1});
            end
                        
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray(vl_nnsiamese(gather(inputs{1}),gather(derOutputs{1})));
            else
                derInputs{1} = vl_nnsiamese(inputs{1},derOutputs{1});
            end
            
            derParams = {};
        end
        
        function obj = siamese(varargin)
            obj.load(varargin);
            obj.loss = 'siam';
        end
    end
end
