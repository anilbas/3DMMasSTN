% Wrapper for vl_nneuclideanloss block

classdef euclidean < dagnn.Loss    
    methods
        function outputs = forward(obj, inputs, ~)
            
            label=inputs{2}(:,1:2,:,:);
            vis=inputs{2}(:,3,:,:);
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                outputs{1} =  gpuArray(vl_nneuclideanloss(gather(inputs{1}), gather(label), gather(vis)));
            else
                outputs{1} =  vl_nneuclideanloss(inputs{1}, label, vis);
            end
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
            
            label=inputs{2}(:,1:2,:,:);
            vis=inputs{2}(:,3,:,:);
            
            useGPU = isa(inputs{1}, 'gpuArray');
            if useGPU
                derInputs{1} = gpuArray(vl_nneuclideanloss(gather(inputs{1}), gather(label), gather(vis), gather(derOutputs{1})));
            else
                derInputs{1} = vl_nneuclideanloss(inputs{1}, label, vis, derOutputs{1});
            end
            
            derInputs{2} = [];
            derParams = {};
        end
        
        function obj = euclidean(varargin)
            obj.load(varargin);
            obj.loss = 'euclidean';
        end
    end
end
