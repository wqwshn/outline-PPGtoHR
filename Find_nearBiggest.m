function [HR,whichPeak] = Find_nearBiggest(Fre, HR_prev, rangeplus,rangeminus)
    %找离HR_prev一定范围内最大的峰位置,前6个里面找最大的
    %Fre：峰位置集合，根据幅值降序排列
    %HR_prev：上一时刻心率
    %range：允许的心率变化范围
       
    HR = HR_prev;
    whichPeak = 0;
    
    if length(Fre) > 5
        len = 5 ;
    else
        len = length(Fre);
    end
    
        for i = 1:len
           if (Fre(i)- HR_prev < rangeplus)  && ( Fre(i) - HR_prev > rangeminus)
               HR = Fre(i);
               whichPeak = i;
               break
           end
        end
%         HR = Fre(1);


end