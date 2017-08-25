% Wrapper for vl_nnsse block

classdef sse < dagnn.Loss
    
    methods
        function outputs = forward(obj, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} =  gpuArray(vl_nnsse(gather(inputs{1})));
            else
                outputs{1} =  vl_nnsse(inputs{1});
            end
                        
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray(vl_nnsse(gather(inputs{1}),gather(derOutputs{1})));
            else
                derInputs{1} = vl_nnsse(inputs{1},derOutputs{1});
            end
            
            derParams = {};
        end
        
        function obj = sse(varargin)
            obj.load(varargin);
            obj.loss = 'sse';
        end
    end
end
