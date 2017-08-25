% Wrapper for vl_nnsplit block
% inputs{1} :   X     : 1 x 1 x 16 x b
% outputs{1}:   y     : 1 x 1 x 10 x b
% outputs{1}:   r     : 1 x 1 x 3 x b
% outputs{1}:   t     : 1 x 1 x 2 x b
% outputs{1}:   s     : 1 x 1 x 1 x b

classdef split < dagnn.Layer
   
    methods
        function outputs = forward(~, inputs, ~)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
            [y,r,t,s] = vl_nnsplit(gather(inputs{1}));
            outputs = {gpuArray(y),gpuArray(r),gpuArray(t),gpuArray(s)};
            else
            [y,r,t,s] = vl_nnsplit(inputs{1});
            outputs = {y,r,t,s};    
            end
            
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
            derInputs{1} = gpuArray( vl_nnsplit(gather(inputs{1}),gather(derOutputs{1}), gather(derOutputs{2}), gather(derOutputs{3}), gather(derOutputs{4})) );
            else
            derInputs{1} = vl_nnsplit(inputs{1},derOutputs{1},derOutputs{2},derOutputs{3},derOutputs{4});    
            end
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(~, inputSizes)
            outputSizes = {1 1 inputSizes{1}(3) inputSizes{1}(4)} ;
        end
        
        function obj = split(varargin)
            obj.load(varargin);
        end
    end
end
