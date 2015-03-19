%%Local visualizations for sparse basis

function [f] = vis_basis(basis,vis_size)

if nargin<2
    vis_size = [size(basis,2)/10,10]
end


for ii = 1:vis_size(1)
    for jj =1:vis_size(2)
        tmp = reshape(basis(:,ii*vis_size(1)+jj),[512,512]);
        subplot(vis_size(1),vis_size(2),ii*vis_size(2)+jj);
        imshow(tmp)
    end
end

end