function [fx,fy] = readFlowFromFile(file)
    f = fopen(file,'rb');
    
    rows = fread(f,1,'int32');
    cols = fread(f,1,'int32');
% 
%     fx = zeros(rows,cols);
%     fy = zeros(rows,cols);

    fall = fread(f,rows*cols*2,'float32');
    fxall = fall(1:2:end);
    fyall = fall(2:2:end);
    
    fx = reshape(fxall,cols,rows).';
    fy = reshape(fyall,cols,rows).';
    
%     for ii = 1:rows
%         for jj = 1:cols
%             fx(ii,jj) = fread(f,1,'float32');
%             fy(ii,jj) = fread(f,1,'float32');
%         end
%     end

    fclose(f);
end