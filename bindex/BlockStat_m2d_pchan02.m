%%Masato 2013b
% sbatch --account=kuang_lab -p huce_intel -J m2d -n 1 -t 480 --mem=20000 -o "slurm" --wrap='\matlab -nodesktop -nodisplay -nosplash -r "BlockStat_m2d_pchan02;exit" >& slurm-${SLURM_JOB_NAME}-${SLURM_JOBID}' --mail-type=END
% TODO 02.m: label contiguous, quick polyfit
% 2 mins in kuang-intel

% To load this blocking index:
%{
ver=['8a14_',caseid];  %
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
PER8a14 = (mat_load2.PER);
PER8a14 = (mat_load2.label>0);
%}
 %load(fn_load2,'timeNan','PER')
% PER = single(PER>=5);  PER(:,:,timeNan)=nan;
% 1=blocking, 0=no-blocking, nan=undefined

%% load and save data
tic;

season=true;
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%caseid=['ERA-interim_19790101-20171231']; season=true;
%verX='x912';
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','ds','Cold_yxht','Hot_yxht','T850f_yxht','areaEarth')
%load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh' ,'ds','yStart','yEnd','nyr' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend','lsm_jja' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend','lsm_djf')
load(['temp_',verX,'_',caseid,'.mat'],'caseid','thresh' ,'ds','yStart','yEnd','nyr','f_h2d','latt42' ,'ds_jja','nd_jja','hJJAstart','hJJAend','lsm_jja' ,'ds_djf','nd_djf','hDJFstart','hDJFend','lsm_djf')

%prm.dphi = 5;  %dphi/2 ~= 14deg (15)
prm.yS1  = 6; % 73.95deg  %8-3; % 68.37deg
prm.yS2  = find(latt42<-thresh{5},1,'last');  %18; % 40.46deg
prm.yN1  = 64+1-prm.yS2;
prm.yN2  = 64+1-prm.yS1;
prm.latyday = 5;  % 14deg (13.5)
prm.lonyday = 6;  % 16.875deg (18)
prm.latonset = 7;  % 19.53deg (20.25)
prm.lononset = 10;  % 28.125deg (27)

prm.A = 0;
%prm.O = 0.5;
%prm.S
%prm.D = 5;

yS1=prm.yS1; yS2=prm.yS2; yN1=prm.yN1; yN2=prm.yN2;

%ver=['8a14_',caseid];  %
ver=[strTitle(1:4),'_',caseid];  %
%strTitle = '8a00: M13';
%strTitle = '8a10: M13, 49-77N in 40-68';
%strTitle = '8a11: M13, 49-77N in 49-77';
%strTitle = '8a12: M13, 40-77N in 40-77';
%strTitle = '8a13: M13, Z500, 40-74N, D=1';
%strTitle = '8a14: M13, Z500, 40-74N, D=5,smth';

 fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
latt42  = ncread(fn_t42,'lat');
lont42  = ncread(fn_t42,'lon');

%% load
%disp('before load'); toc
 %load(['Z500_06xx',caseid,'.mat'],'ZaDaily','Z500Daily','x','y','Z500sd');  % given in bindex_pre_intp_pchan.m
ver0=ver; ver0(1)='0'; ver0(3:4)='xx';
load(['../index_wise/Var_',verX,ver0,'.mat'],'Wgt','Wgt_jja','Wgt_djf','VarDaily','time');  % TODO best given in BlockStat_anomaly_pchan01.m
disp('finish load'); toc

 % ~/sw/ecmwfapi/wget.ecmwf.txt
 % ~/sw/ecmwfapi/interim-z500.py -> zg_day_caseid.nc
 % ncap2 -O -s "z@scale_factor=float(z@scale_factor)/9.81f; z@add_offset=float(z@add_offset)/9.81f;" zg_day_ERA-interim_19790101-20151231.nc nco_z500_zg_day_ERA-interim_19790101-20151231.nc
 % ln -s   ../sks
 % bindex_pre_intp_pchan -> ../sks/int_z500_zg_day_',caseid,'N.nc
 % this matlab
%load(['Z500_',caseid,'.mat'],'Z500Daily');
 %fn_int   = ['../sks/int_z500_zg_day_',caseid,'N.nc'];

%{
 % ~/sw/ecmwfapi/wget.ecmwf.txt: ncap2 -s "z@scale_factor=float(z@scale_factor)/9.81f; z@add_offset=float(z@add_offset)/9.81f;" z250_00Z_ERA-interim_19790101-20151231.nc nco_z250_00Z_ERA-interim_19790101-20151231.nc
fn_nco   = ['../ERA-interim/nco_z500_zg_day_',caseid,'.nc'];  % 2a
%fn_nco   = ['../ERA-interim/nco_z250_00Z_',caseid,'.nc'];  % 2b
VarDaily = single(ncread(fn_nco,'z'));
VarDaily = squeeze(VarDaily);
disp('finish load'); toc
%ds = size(VarDaily);  % load from temp.mat

%% check lat lon, referencing fn_t42
time = ncread(fn_nco,'time');
lat  = ncread(fn_nco,'latitude');
lon  = ncread(fn_nco,'longitude');

if (lat(2)<lat(1))
  VarDaily = VarDaily(:,end:-1:1 ,:);
  lat = lat(end:-1:1);
end
if (max(abs(latt42(1:end)-lat))>0.1 || max(abs(lont42-lon))>0.1)
  disp('error'); return;
end
%}

%% nan, sign, angle, demean, movmean
%  B = movmean(VarDaily,[0 prm.dphi],2) -movmean(VarDaily,[prm.dphi 0],2);  % canceling at phi0
  Wgt = nan(ds,'single');
%  Wgt(:,[yS1:yS2 yN1:yN2],:) = B(:,[yS1:yS2 yN1:yN2],:);

%  VarDaily = movmean(VarDaily,[0 prm.dphi],2);
%  VarDaily = movmean(VarDaily,6,2);
  VarDaily = movmean(VarDaily,prm.dphi,2);
  Wgt(:,[yN1:yN2],:) = VarDaily(:,yN1:yN2,:)-VarDaily(:,[yN1:yN2]-prm.dphi,:);
%  Wgt(:,[yN1:yN2]-3,:) = VarDaily(:,yN1:yN2,:)-VarDaily(:,[yN1:yN2]-prm.dphi,:);  %8a10
%  Wgt(:,[yN1+0:yN2+0],:) = VarDaily(:,yN1+3:yN2+3,:)-VarDaily(:,[yN1+3:yN2+3]-prm.dphi,:);
  clear VarDaily B;

%timeNan = ds(3);  % 700?, 702?, 703?
%VarDaily(:,:,timeNan) = nan;

%todo  Wgt = Wgt .* repmat(reshape(sign(latt42), [1 ds(2) 1]), [ds(1) 1 ds(3)]);

%  Wgt = -Wgt;  % 7060
%  Wgt = Wgt*cosd(prm_angle) + U850Daily*sind(prm_angle);  % 7060

%  wrk  = nanmean( reshape(permute(double(Wgt),[2 1 3]), ds(2),[]) ,2); %xyt->yxt, double precision needed
%  Wgt = Wgt - repmat( reshape(wrk, [1 ds(2) 1]), [ds(1) 1 ds(3)]);

% 7002, 70?0
%  Wgt = movmean(Wgt,prm_avg0,3,'Endpoints','fill');  % 7002, mimic integration  'includenan',
%  Wgt = movmean(Wgt,prm_avg,3,'Endpoints','fill');  % mimic extreme definition
%Wgt = filter([0.5 ones(1,7) 0.5], 8, Wgt, [], 3);% Though Schwierz remove climatology first..
%Wgt(:,:,5:end-4) = Wgt(:,:,9:end);
%Wgt(:,:,[1:4,end-3:end]) = nan;

%% collect JJA
%{
Wgt_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  Wgt_jja(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
Wgt_djf = zeros(ds_djf,'single');
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  Wgt_djf(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
WgtOrg = Wgt; clear Wgt

%% remove trend
Wgtjja_xyn = squeeze(mean( reshape(Wgt_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr

Wgtdjf_xyn = squeeze(mean( reshape(Wgt_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr

Wgt_jja = Wgt_jja - reshape(repmat(reshape(movmean(Wgtjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);

Wgt_djf = Wgt_djf - reshape(repmat(reshape(movmean(Wgtdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);

% no quantile

% no land

%% write back to full year data for fortran
Wgtjja_smth = movmean(Wgtjja_xyn,5,3);
Wgtdjf_smth = movmean(Wgtdjf_xyn,5,3);
%Wgt_jja = Wgt_jja - reshape(repmat(reshape(movmean(Wgtjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
%mn2t_djf = mn2t_djf - reshape(repmat(reshape(movmean(mn2t_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);

Wgt = nan(ds,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  Wgt(:,:,tstart-40:tend+40) = WgtOrg(:,:,tstart-40:tend+40) -repmat(Wgtjja_smth(:,:,yyyy-yStart+1), [1 1 ds_jja(3)/nyr+80]);
%  Wgt_jja(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);  % 'Wgt_jja' is reused here  Done in "remove trend"

  tpointer = tpointer +tend-tstart+1;
end

tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  Wgt(:,:,tstart-40:tend+40) = WgtOrg(:,:,tstart-40:tend+40) -repmat(Wgtdjf_smth(:,:,yyyy-yStart), [1 1 ds_djf(3)/(nyr-1)+80]);
%  Wgt_djf(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);  % 'Wgt_djf' is reused here

  tpointer = tpointer +tend-tstart+1;
end
clear WgtOrg
%}

PER = Wgt>0;

%% tracking algorithm
% see Blocklabel
%  F_id = repmat(reshape(1:128*64, [128 64 1]), [1 1 days]);
%  F_id = F_id.*(PER==1);

if (prm.D>1)
  nBlock = 0; xyBlock =[];  % txyltxy
  xyBlock_yday=[];
  WgtMax = Wgt.*(Wgt>0);
  for t=1:ds(3)
    wrk = WgtMax(:,:,t);
    sumPast = 0;
    while nansum(wrk(:)) > sumPast
        sumPast = nansum(wrk(:));
        wrk = max( cat(4, wrk, circshift(wrk,[1,0]),circshift(wrk,[-1,0]),circshift(wrk,[0,1]),circshift(wrk,[0,-1]), ...
                        circshift(wrk,[1,1]),circshift(wrk,[1,-1]),circshift(wrk,[-1,1]),circshift(wrk,[-1,-1]) ), [], 4);
% circshift in y could mess up things. Make sure nothing cross the pole
%        wrk = movmax(wrk,3,1);  % circshift!!
%        wrk = movmax(wrk,3,2);
        wrk(~(Wgt(:,:,t)>0)) = 0;
    end
    WgtMax(:,:,t) = wrk;

    xyBlock_tday = [];
    [row,col] = find(WgtMax(:,:,t)==Wgt(:,:,t));  % today
    for m = 1:size(xyBlock_yday,2)
      if (~isempty(row))
        dist = [ (-ds(1)/2+mod(xyBlock_yday(2,m)-row'+ds(1)/2, ds(1))); (xyBlock_yday(3,m)-col') ];
        bDist = abs(dist(1,:))<=prm.lonyday & abs(dist(2,:))<=prm.latyday;
        dist(:,~bDist) = nan;
      else
        bDist =[];
      end

      if (sum(bDist)==0)  % finish becoz criteria 2
        if ( (t-xyBlock_yday(5,m))<prm.D )
          % clean all
          for tt = xyBlock_yday(5,m):t-1
            mm = find( xyBlock(4,:)==xyBlock_yday(4,m) & xyBlock(1,:)==tt );
            ii = xyBlock(2,mm); jj = xyBlock(3,mm);
            wrk = WgtMax(:,:,tt);
            wrk( wrk==wrk(ii,jj)) = 0;
            WgtMax(:,:,tt) = wrk;
            xyBlock(:,mm) = [];
          end
        end  % nothing to be done if D>=5?
      else  % find closest 'today'
        [~,mDist] = min(sum(dist.^2,1),[],2);
        % compare onset
        if (abs( -ds(1)/2+mod(xyBlock_yday(6,m)-row(mDist)+ds(1)/2, ds(1)) )<=prm.lononset && abs( xyBlock_yday(7,m)-col(mDist) )<=prm.latonset )
          temp = [t; row(mDist); col(mDist); xyBlock_yday(4:7,m)];
          xyBlock_tday = [xyBlock_tday temp];
          xyBlock      = [xyBlock      temp];
          row(mDist)=[]; col(mDist)=[];  % clean: first come first serve
        else  % finish becoz criteria 3
          if ( (t-xyBlock_yday(5,m))<prm.D )
            % clean all
            for tt = xyBlock_yday(5,m):t-1
              mm = find( xyBlock(4,:)==xyBlock_yday(4,m) & xyBlock(1,:)==tt );
              ii = xyBlock(2,mm); jj = xyBlock(3,mm);
              wrk = WgtMax(:,:,tt);
              wrk( wrk==wrk(ii,jj)) = 0;
              WgtMax(:,:,tt) = wrk;
              xyBlock(:,mm) = [];
            end
          end  % nothing to be done if D>=5?
          % create onset for new one later
        end
      end
    end % loop m for yday

    for m=1:length(row)  % not matched
      nBlock = nBlock+1;
      id = nBlock;
      xyBlock_tday = [xyBlock_tday [t; row(m); col(m); id; t; row(m); col(m)]];
      xyBlock      = [xyBlock      [t; row(m); col(m); id; t; row(m); col(m)]];
    end
    xyBlock_yday = xyBlock_tday; xyBlock_tday=[];
  end % loop t

  Wgt = Wgt.*(WgtMax>0);
  PER = Wgt>0;
% F_id
  timeNan = [1:(prm.D-1), (ds(3)-prm.D+2):ds(3)];
  % length(unique(xyBlock(4,:)))  % number of blocks tracked
%  clear WgtMax   % for label
end  % D>1

toc
%% quality check
% meanZ500 in bindex_pre_intp_pchan
% meanJJA in xtrm_reanalysis_pchan02.m
addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%latt42=y; lont42=x;
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added
rng(2017);

fn_figure = ['../index_wise/all',ver,'.ps'];
%system(['rm ',fn_figure]);

%{
% Wgtjja_xyn jjaQuantile Wgt_jja

% calulate trend for comparison with Horton et al. 2015 figure 1
% xyn trend
%
Weight_t = 1:nyr;
Weight_t = Weight_t - mean(Weight_t);
Weight_t = Weight_t / sumsqr(Weight_t);
Wgtjja_trend = sum(Wgtjja_xyn.*repmat(reshape(Weight_t,[1 1 nyr]),[ds(1:2) 1]),3);

Weight_t = 1:nyr-1;
Weight_t = Weight_t - mean(Weight_t);
Weight_t = Weight_t / sumsqr(Weight_t);
Wgtdjf_trend = sum(Wgtdjf_xyn.*repmat(reshape(Weight_t,[1 1 nyr-1]),[ds(1:2) 1]),3);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
%axesm('MapProjection','vperspec','origin',[90 0],'MapParallels',6000,'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','ortho','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','eqdazim','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','breusing','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
contourfm(latt42,lonplot,double(Wgtjja_trend([1:end,1],:)'),[-4:4]*0.5);  % TODO
%pcolormPC(latt42,lont42, Wgtjja_trend'); shading flat;
%contourm(latt42,lonplot,double(Wgtjja_trend([1:end,1],:)'),[-3:3]*0.5); %axis equal tight; %,'showtext','on'
%contourm(latt42,lonplot,double(Wgtjja_trend([1:end,1],:)'),'showtext','on'); %axis equal tight; %,'showtext','on'
%colormap(jet(12)); caxis([0 12]); colorbar;
colormap(b2rPC(-5*0.5, 5*0.5, 10)); colorbar;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitle,'JJA Wgt trends (1979-2017) (unit/yr)'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
contourfm(latt42,lonplot,double(Wgtdjf_trend([1:end,1],:)'),[-4:4]*0.5);
%pcolormPC(latt42,lont42, Wgtdjf_trend'); shading flat;
%contourm(latt42,lonplot,double(Wgtdjf_trend([1:end,1],:)'),[-0.06:0.02:0.06]); %axis equal tight;
%colormap(jet(12)); caxis([0 12]); colorbar;
colormap(b2rPC(-5*0.5, 5*0.5, 10)); colorbar;
plotm(coastlat,coastlon,'k')
title({'DJF Wgt trends (1979/80-2014/15) (unit/yr)'},'fontsize',16);
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);
%

% xyn: mean JJA/DJF time series for ramdom points: check movmean 5
jArr = find(latt42>0); jArr = jArr(1:2:end); %set(groot,'defaultAxesColorOrder',hsv(length(jArr)));
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);

subplot(1,2,1); hold on;
xlabel({'year'}); ylabel('Wgt (unit): each line shifted by 100');  % TODO
ax=gca; ax.ColorOrder = hsv(length(jArr));
%ax.ColorOrder = lines(length(jArr));
%for j=jArr(:)'
for jj = 1:length(jArr)
  j = jArr(jj); 
%  nLand = sum(lsm_jja(:,j));
%  if (nLand>0) 
%    iArr = find(lsm_jja(:,j));
%    i = iArr(randi(nLand));
    i = randi(ds(1));
    ax.ColorOrderIndex = jj;
%    plot(yStart:yEnd, squeeze(Wgtjja_xyn(i,j,:)));
%    ax.ColorOrderIndex = jj;
%    plot(yStart:yEnd, movmean( squeeze(Wgtjja_xyn(i,j,:)), 5), '--');
    plot(yStart:yEnd, jj*100-mean(Wgtjja_xyn(i,j,1:3)) +squeeze(Wgtjja_xyn(i,j,:)));
    ax.ColorOrderIndex = jj;
    plot(yStart:yEnd, jj*100-mean(Wgtjja_xyn(i,j,1:3)) +movmean( squeeze(Wgtjja_xyn(i,j,:)), 5), '--');
%  end  % TODO land or not
end
title({strTitle,'mean Wgt in JJA'});
xlim([yStart yEnd]);

subplot(1,2,2); hold on;
xlabel({'year'}); ylabel('Wgt (unit): each line shifted by 100');
ax=gca; ax.ColorOrder = hsv(length(jArr));
%ax.ColorOrder = lines(length(jArr));
%for j=jArr(:)'
for jj = 1:length(jArr)
  j = jArr(jj); 
%  nLand = sum(lsm_djf(:,j));
%  if (nLand>0) 
%    iArr = find(lsm_djf(:,j));
%    i = iArr(randi(nLand));
    i = randi(ds(1));
    ax.ColorOrderIndex = jj;
%    plot(yStart+1:yEnd, squeeze(Wgtdjf_xyn(i,j,:)));
%    ax.ColorOrderIndex = jj;
%    plot(yStart+1:yEnd, movmean( squeeze(Wgtdjf_xyn(i,j,:)), 5), '--');
    plot(yStart+1:yEnd, jj*100-mean(Wgtdjf_xyn(i,j,1:3)) +squeeze(Wgtdjf_xyn(i,j,:)));
    ax.ColorOrderIndex = jj;
    plot(yStart+1:yEnd, jj*100-mean(Wgtdjf_xyn(i,j,1:3)) +movmean( squeeze(Wgtdjf_xyn(i,j,:)), 5), '--');
%  end
end
title('mean Wgt in DJF');

print(gcf, '-dpsc2','-append',fn_figure);

%% plot quantile xtrm_reanalysis_pchan02.m / xtrmfreq
%{
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

jjaQuantile(~lsm_jja) = nan;
%contourfm(latt42,lonplot,double(jjaQuantile([1:end,1],:)'),[-2:0.5:2]);
pcolormPC(latt42,lont42,jjaQuantile'); shading flat;
colormap(jet(12)); caxis([0 12]); colorbar;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitle,'JJA 99th percentile of Wgt'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

djfQuantile(~lsm_djf) = nan;
%contourfm(latt42,lonplot,double(djfQuantile([1:end,1],:)'),[-2:0.5:2]);
pcolormPC(latt42,lont42,djfQuantile'); shading flat;
colormap(jet(12)); caxis([0 24]); colorbar;
plotm(coastlat,coastlon,'k')
title({'DJF 99th percentile of Wgt'},'fontsize',16);
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);
%}

%% SeasonalCycle, Wgt, xtrm_reanalysis
%
rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%jArr = find(latt42>0);
jArr = [yN1:yN2]';
jArr = jArr(1:2:end);
for jj = 1:length(jArr)
  j = jArr(jj);
%  nLand = sum(lsm_jja(:,j));
%  if (nLand>0)
%    iArr = find(lsm_jja(:,j));
%    i = iArr(randi(nLand));
    i = randi(ds(1));
    subplot(3,5,jj); hold on;
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr);
    plot(reshape( squeeze(Wgt_jja(i,j,:)),[],nyr ));
    plot(mean(reshape( squeeze(Wgt_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
    plot([1 nd_jja],prm.A*[1 1],'k-');  % TODO
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E']);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
%  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);
%print(gcf,'-dpdf','-r600',['SeasonalCycleJJA_',textTH,'_',text,'.pdf'])

rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%jArr = find(latt42>0);
jArr = [yN1:yN2]';
jArr = jArr(1:2:end);
for jj = 1:length(jArr)
  j = jArr(jj);
%  nLand = sum(lsm_djf(:,j));
%  if (nLand>0)
%    iArr = find(lsm_djf(:,j));
%    i = iArr(randi(nLand));
    i = randi(ds(1));
    subplot(3,5,jj); hold on;
    ax=gca; ax.ColorOrder = jet(nyr-1);
    plot(reshape( squeeze(Wgt_djf(i,j,:)),[],nyr-1 ));
    plot(mean(reshape( squeeze(Wgt_djf(i,j,:)),[],nyr-1 ),2),'k','linewidth',1);
    plot([1 90],prm.A*[1 1],'k-');
    title([num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E']);
    axis tight;
%  end
end
print(gcf, '-dpsc2','-append',fn_figure);
%print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])

%system(['ps2pdf ',fn_figure]);
toc
%}

%% create nc for program  % see sch*.m, bindex_pre_intp_pchan.m
%% call fortran program

% output
%{
save(['../index_wise/BlockFreq_',ver,'.mat'],'prm','timeNan','PER','Wgt','lont42','latt42','-v7.3')

 fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
fn_savenc = ['../index_wise/BlockFreq_',ver,'.nc'];
 system(['rm ',fn_savenc]); 
system(['ncks -6 -v lat,lat_bnds,lon,lon_bnds ',fn_t42,' ',fn_savenc]);
%nccreate(fn_savenc,'prm_avg','DataType','single','Format','64bit')
%nccreate(fn_savenc,'prm_prct','DataType','single')
%nccreate(fn_savenc,'prm_abs','DataType','single')
nccreate(fn_savenc,'timeNan','Dimensions',{'dim5',numel(timeNan)},'DataType','single')
nccreate(fn_savenc,'PER','Dimensions',{'lon',128,'lat',64,'time',ds(3)},'DataType','int8')
ncwriteatt(fn_savenc,'PER','description',ver)
%ncwrite(fn_savenc,'prm_avg',prm_avg)
%ncwrite(fn_savenc,'prm_prct',prm_prct)
%ncwrite(fn_savenc,'prm_abs',prm_abs)
ncwrite(fn_savenc,'timeNan',timeNan)
ncwrite(fn_savenc,'PER',int8(PER))
%system(['ln -s ',fn_save,' BlockFreq_2222_',caseid,'.mat']);
%}

%% quick polyfit, lagcorr
%
 %text='AOefbCV1HF'; thresh={0.01,'quantile',5,5,0}; caseid=[text,'T63h00'];
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
 %load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','ds','Cold_yxht','Hot_yxht','T850f_yxht','areaEarth')
 %load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','yStart','yEnd','nyr' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend')
load(['temp_',verX,'_',caseid,'.mat'],'caseid','strTitleX','thresh' ,'ds','yStart','yEnd','nyr','f_h2d','fn_t42','latt42','areaEarth' ,'ds_jja','nd_jja','Hot_jja','mx2t_jja','HotQuantile','hJJAstart','hJJAend','lsm_jja','Hotid_jja','HotAttr' ,'ds_djf','nd_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend','lsm_djf','Coldid_djf','ColdAttr' ,'Hot_xyn')

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

%    lat_bnds = ncread(fn_t42,'lat_bnds');
%    R = 6371; %km
 %    areaEarth = (2*pi)*(R^2)* (cosd(lat1a') *1*pi/180);
%    areaEarth = (2*pi)*(R^2)* (sind(lat_bnds(2,:))-sind(lat_bnds(1,:)));
 %    areaEarth = areaEarth(1:floor(ds_jja(2)/2));
%  areaEarth(latt42(:)<=thresh{5})=0;

%timeNan = unique([timeNan, 1:(thresh{3}-1)/2,ds(3)-(thresh{3}-1)/2+1:ds(3)]);

%Wgt = T850Adv3d; % 7002, 70?0
%Wgt_jja = mx2t_jja;
%Wgt_djf = mn2t_djf;
%PER_jja = Hot_jja;
%PER_djf = Cold_djf;

%timeNan(timeNan>ds(3)) = [];

%% collect JJA
%
timeNan = [];

PER_jja = false(ds_jja);
Wgt_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  PER_jja(:,:,tpointer+(0:tend-tstart)) = PER(:,:,tstart:tend);
  Wgt_jja(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
PER_djf = false(ds_djf);
Wgt_djf = zeros(ds_djf,'single');
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  PER_djf(:,:,tpointer+(0:tend-tstart)) = PER(:,:,tstart:tend);
  Wgt_djf(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
clear PER Wgt

Wgt_jja(isnan(Wgt_jja))=0;
Wgt_djf(isnan(Wgt_djf))=0;

%%
mx2t_jja = mx2t_jja - repmat(max(HotQuantile,thresh{4}),[1 1 ds_jja(3)]);
%Wgt_jja = Wgt_jja - prm.A* repmat([sdJJA_SH*ones(1,ds(2)/2), sdJJA_NH*ones(1,ds(2)/2)],[ds(1) 1 ds_jja(3)]);
%Wgt_jja(:) = 1; %TODO

Hot_n = [areaEarth * squeeze(mean(mean(reshape(Hot_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
Hotw_n = [areaEarth * squeeze(mean(mean(reshape(Hot_jja.*mx2t_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERwjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*Wgt_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERljja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*lsm_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERlwjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*lsm_jja.*Wgt_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
%disp(sprintf('jja: D=%i GHGNT=%i %+.3f %+.3f %+.3f %+.3f',prm.D,prm.GHGNT, corr(PERjja_n(:),Hot_n(:)), corr(PERljja_n(:),Hot_n(:)) , corr(PERwjja_n(:),Hotw_n(:)), corr(PERlwjja_n(:),Hotw_n(:)) ));
disp(sprintf('jja: D=%i dphi=%i %+.3f %+.3f %+.3f %+.3f',prm.D,prm.dphi, corr(PERjja_n(:),Hot_n(:)), corr(PERljja_n(:),Hot_n(:)) , corr(PERwjja_n(:),Hotw_n(:)), corr(PERlwjja_n(:),Hotw_n(:)) ));

wrkar1 = corr(Hot_n(2:end),Hot_n(1:end-1));
wrkar2 = corr(PERljja_n(2:end),PERljja_n(1:end-1));
wrkar3 = corr(PERjja_n(2:end),PERjja_n(1:end-1));
%disp(sprintf('autocorr H:%+.3f Pl:%+.3f P:%+.3f tstar/t:%.3f %.3f',wrkar1,wrkar2,wrkar3,(1-wrkar1*wrkar2)/(1+wrkar1*wrkar2) ,(1-wrkar1*wrkar3)/(1+wrkar1*wrkar3) ));
wrkar1 = corr(Hotw_n(2:end),Hotw_n(1:end-1));
wrkar2 = corr(PERlwjja_n(2:end),PERlwjja_n(1:end-1));
wrkar3 = corr(PERwjja_n(2:end),PERwjja_n(1:end-1));
%disp(sprintf('autocorr H:%+.3f Pl:%+.3f P:%+.3f tstar/t:%.3f %.3f',wrkar1,wrkar2,wrkar3,(1-wrkar1*wrkar2)/(1+wrkar1*wrkar2) ,(1-wrkar1*wrkar3)/(1+wrkar1*wrkar3) ));

rng(2017);
%bootci(1000,{@corr,Hot_n,PERjja_n})
%bootr00 = bootstrp(1000,@corr,Hot_n,PERjja_n);
rng(2017);
%bootci(1000,{@corr,Hot_n,PERljja_n})
%bootr01 = bootstrp(1000,@corr,Hot_n,PERljja_n);
rng(2017);
%bootci(1000,{@(x,y1,y0) corr(x,y1)-corr(x,y0),Hot_n,PERljja_n,PERjja_n})
%bootd03 = bootstrp(1000,@(x,y1,y0) corr(x,y1)-corr(x,y0),Hot_n,PERljja_n,PERjja_n);
%max(abs(bootr01-bootr00-bootd03))

%mean(bootstrp(1000,@(x,y1,y0) corr(x,y1)-corr(x,y0),Hot_n,PERljja_n,PERjja_n)<=0)
%quantile(bootr1-randsample(bootr2,1000),[0.025 0.975])
%mean(bootr1-randsample(bootr2,1000)<=0)  % random pairing

%disp([textTH,'_',ver,':  ',num2str(hotstat(3))])
%system(['echo ',textTH,'_',ver,':  ',num2str(hotstat(3)), ' >> corr-hot']);

% fn_save  = ['../index_wise/scatter_',textTH,'_',ver,'.mat'];
% save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');
%save(['scatter_',ver,'.mat'],'ver','strTitle','prm','timeNan','lont42','latt42', 'Hot_n','Hotw_n','PERjja_n','PERwjja_n','PERljja_n','PERlwjja_n', 'sdJJA_NH','sdJJA_SH','sdDJF_NH','sdDJF_SH','-v7.3');
toc

%{
%for prmd = 1:2:13
%  prm.D = prmd;
for pm = 1:8
  DArr = [1:2:7, 10:4:22];
  prm.D = DArr(pm);
  for prma = -15:5:0
    prm.GHGNT = prma;
    clearvars -except  prm rrr pm;
    BlockStat_scherrer_pchan05;  % TODO
    rrr.r00( pm, (prm.GHGNT+20)/5 ) = corr(PERjja_n(:),Hot_n(:));
    rrr.r01( pm, (prm.GHGNT+20)/5 ) = corr(PERljja_n(:),Hot_n(:));
    rrr.r02( pm, (prm.GHGNT+20)/5 ) = corr(PERwjja_n(:),Hotw_n(:));
    rrr.r03( pm, (prm.GHGNT+20)/5 ) = corr(PERlwjja_n(:),Hotw_n(:));
  end
end
% fn_save  = ['corrTab_',verX,ver,'.mat'];
% save(fn_save,'prm','rrr','strTitle');
addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(-15:5:0, 1:8,rrr.r02);
%colormap(gca,flipud(hot)); colorbar;
colormap(gca,b2r(-1,1)); colorbar;
  for pm = 1:8
    for prma = -15:5:0
      text(prma,pm,sprintf('%+.3f',rrr.r02(pm,(prma+20)/5)),'HorizontalAlignment','center','fontsize',16);
    end
  end
  title({strTitle},'FontSize',20); yticks(1:8); yticklabels([1:2:7, 10:4:22]);
%xticks(1:np); xticklabels(strTickLabel); xtickangle(45);
xlabel('GHGNT'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(-15:5:0, 1:8,rrr.r03);
%colormap(gca,flipud(hot)); colorbar;
colormap(gca,b2r(-1,1)); colorbar;
  for pm = 1:8
    for prma = -15:5:0
      text(prma,pm,sprintf('%+.3f',rrr.r03(pm,(prma+20)/5)),'HorizontalAlignment','center','fontsize',16);
    end
  end
  title({strTitle},'FontSize',20); yticks(1:8); yticklabels([1:2:7, 10:4:22]);
%xticks(1:np); xticklabels(strTickLabel); xtickangle(45);
xlabel('GHGNT'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);
%}

%
%% label contiguous, see blocklabel, m2d
if (exist('rrr','var')==0)
% method array
%PERid_jja = repmat(reshape([1:ds(1)*floor(ds(2)/2)*nd_jja], [ds(1) floor(ds(2)/2) nd_jja]), [1 2 nyr]);
PERid_jja = reshape([1:prod(ds_jja)], [ds_jja]);
PERid_jja = PERid_jja.*PER_jja;
wrk = zeros(ds(1),floor(ds(2)/2)+1,nd_jja+1);
for yyyy = yStart:yEnd
  wrk(:,1:end-1,1:end-1) = PERid_jja(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart)*nd_jja+(1:nd_jja));
  bwrk = wrk>0;
  sumPast = 0;
  while sum(wrk(:)) > sumPast
    sumPast = sum(wrk(:));
    wrk = max( cat(4, wrk, circshift(wrk,1,1),circshift(wrk,-1,1)), [], 4);
    wrk = max( cat(4, wrk, circshift(wrk,1,2),circshift(wrk,-1,2)), [], 4);
    wrk = max( cat(4, wrk, circshift(wrk,1,3),circshift(wrk,-1,3)), [], 4);
%    wrk = max( cat(4, wrk, circshift(wrk,[1,0]),circshift(wrk,[-1,0]),circshift(wrk,[0,1]),circshift(wrk,[0,-1]), ...
%                    circshift(wrk,[1,1]),circshift(wrk,[1,-1]),circshift(wrk,[-1,1]),circshift(wrk,[-1,-1]) ), [], 4);
    wrk(~bwrk) = 0;
  end
  PERid_jja(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart)*nd_jja+(1:nd_jja)) = wrk(:,1:end-1,1:end-1);
end
PERid_jja(:,1:floor(ds(2)/2),:) = 0;
%PERid_jja(:,latt42(:)<=thresh{5},:) = 0;  % TODO 20 vs. 40 lat

PERid_djf = reshape([1:prod(ds_djf)], [ds_djf]);
PERid_djf = PERid_djf.*PER_djf;
wrk = zeros(ds(1),floor(ds(2)/2)+1,nd_djf+1);
for yyyy = yStart+1:yEnd
  wrk(:,1:end-1,1:end-1) = PERid_djf(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart-1)*nd_djf+(1:nd_djf));
  bwrk = wrk>0;
  sumPast = 0;
  while sum(wrk(:)) > sumPast
    sumPast = sum(wrk(:));
    wrk = max( cat(4, wrk, circshift(wrk,1,1),circshift(wrk,-1,1)), [], 4);
    wrk = max( cat(4, wrk, circshift(wrk,1,2),circshift(wrk,-1,2)), [], 4);
    wrk = max( cat(4, wrk, circshift(wrk,1,3),circshift(wrk,-1,3)), [], 4);
%    wrk = max( cat(4, wrk, circshift(wrk,[1,0]),circshift(wrk,[-1,0]),circshift(wrk,[0,1]),circshift(wrk,[0,-1]), ...
%                    circshift(wrk,[1,1]),circshift(wrk,[1,-1]),circshift(wrk,[-1,1]),circshift(wrk,[-1,-1]) ), [], 4);
    wrk(~bwrk) = 0;
  end
  PERid_djf(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart-1)*nd_djf+(1:nd_djf)) = wrk(:,1:end-1,1:end-1);
end
PERid_djf(:,1:floor(ds(2)/2),:) = 0;
%PERid_djf(:,[1:prm.yN1-1,prm.yN2+1:end],:) = 0;
disp('finish label'); toc

clear wrk bwrk

%
PERid_jja = categorical(PERid_jja);
PERid_jja = removecats(PERid_jja,'0');
PERid_jja = renamecats(PERid_jja, cellstr(num2str([1:numel(categories(PERid_jja))]')) );
PERjjaAttr.id = categories(PERid_jja);
nCat = numel(PERjjaAttr.id);
wrk_ytc = permute(single(countcats(PERid_jja,1)),[2 3 1]); %ytc
area_tc = reshape(areaEarth/ds(1) * wrk_ytc(:,:),[ds_jja(3) nCat]);
disp('a5'); toc
PERjjaAttr.area = sum(area_tc,1)';
[~,PERjjaAttr.tmax] = max(area_tc',[],2);
PERjjaAttr.yr = ceil(PERjjaAttr.tmax/nd_jja) +yStart-1;
PERjjaAttr.tstart = zeros([nCat 1],'single');
PERjjaAttr.tend = zeros([nCat 1],'single');

wrk_xyc = reshape( countcats(PERid_jja,3), ds(1)*ds(2),nCat);
F_y      = repmat(reshape(1:ds(2), [1 ds(2)]), [ds(1) 1]) *diag(areaEarth)/ds(1);
F_repmat = repmat(reshape(1:ds(1), [ds(1) 1]), [1 ds(2)]);
F_sin    = sin( F_repmat*2*pi/ds(1)) *diag(areaEarth);  % new, weight by area
F_cos    = cos( F_repmat*2*pi/ds(1)) *diag(areaEarth);
PERjjaAttr.x = atan2(F_sin(:)'*wrk_xyc, F_cos(:)'*wrk_xyc)'/2/pi*ds(1);
PERjjaAttr.x = 0.5 + mod(PERjjaAttr.x-0.5, ds(1));  % range from 0.5 - ds(1).5
PERjjaAttr.x = 0.5 + mod(PERjjaAttr.x-0.5, ds(1));  % bug
PERjjaAttr.y = (F_y(:)'*wrk_xyc)'./PERjjaAttr.area;
wrk_lsm = lsm_jja *diag(areaEarth)/ds(1);
PERjjaAttr.arealsm = (wrk_lsm(:)'*wrk_xyc)';

wrk_xct = countcats(PERid_jja,2);
PERjjaAttr.xt = reshape(atan2(F_sin(:,end)'*wrk_xct(:,:), F_cos(:,end)'*wrk_xct(:,:)),[nCat ds_jja(3)])'/2/pi*ds(1);
PERjjaAttr.xt = 0.5 + mod(PERjjaAttr.xt-0.5, ds(1));  % range from 0.5 - ds(1).5
PERjjaAttr.xt = 0.5 + mod(PERjjaAttr.xt-0.5, ds(1));  % bug
PERjjaAttr.yt = reshape(F_y(1,:)*wrk_ytc(:,:),[ds_jja(3) nCat])./area_tc;
PERjjaAttr.xt(isnan(PERjjaAttr.yt)) = nan;
%PERjjaAttr.ut = %TODO
disp('before loop'); toc

PERjjaAttr.areat = cell([nCat 1]);
for nBlock = 1:nCat
  PERjjaAttr.areat{nBlock} = area_tc(area_tc(:,nBlock)~=0, nBlock);
%  PERjjaAttr.tstart(nBlock) = find(area_tc(:,nBlock),1);
%  PERjjaAttr.tend(nBlock) = find(area_tc(:,nBlock),1,'last');
  PERjjaAttr.tstart(nBlock) = find(any(wrk_ytc(:,:,nBlock),1),1);
  PERjjaAttr.tend(nBlock) = find(any(wrk_ytc(:,:,nBlock),1),1,'last');
end

PERid_djf = categorical(PERid_djf);
PERid_djf = removecats(PERid_djf,'0');
if (any(PER_djf(:)))
PERid_djf = renamecats(PERid_djf, cellstr(num2str([1:numel(categories(PERid_djf))]')) );
PERdjfAttr.id = categories(PERid_djf);
nCat = numel(PERdjfAttr.id);
wrk_ytc = permute(single(countcats(PERid_djf,1)),[2 3 1]); %ytc
area_tc = reshape(areaEarth/ds(1) * wrk_ytc(:,:),[ds_djf(3) nCat]);
disp('b5'); toc
PERdjfAttr.area = sum(area_tc,1)';
[~,PERdjfAttr.tmax] = max(area_tc',[],2);
PERdjfAttr.yr = ceil(PERdjfAttr.tmax/nd_djf) +yStart;
PERdjfAttr.tstart = zeros([nCat 1],'single');
PERdjfAttr.tend = zeros([nCat 1],'single');

wrk_xyc = reshape( countcats(PERid_djf,3), ds(1)*ds(2),nCat);
%F_y      = repmat(reshape(1:ds(2), [1 ds(2)]), [ds(1) 1]) *diag(areaEarth)/ds(1);
%F_repmat = repmat(reshape(1:ds(1), [ds(1) 1]), [1 ds(2)]);
%F_sin    = sin( F_repmat*2*pi/ds(1)) *diag(areaEarth);  % new, weight by area
%F_cos    = cos( F_repmat*2*pi/ds(1)) *diag(areaEarth);
PERdjfAttr.x = atan2(F_sin(:)'*wrk_xyc, F_cos(:)'*wrk_xyc)'/2/pi*ds(1);
PERdjfAttr.x = 0.5 + mod(PERdjfAttr.x-0.5, ds(1));  % range from 0.5 - ds(1).5
PERdjfAttr.x = 0.5 + mod(PERdjfAttr.x-0.5, ds(1));  % bug
PERdjfAttr.y = (F_y(:)'*wrk_xyc)'./PERdjfAttr.area;
wrk_xyc = reshape( countcats(circshift(PERid_djf,[11 -5 0]),3), ds(1)*ds(2),nCat);  %TODO
wrk_lsm = lsm_djf *diag(areaEarth)/ds(1);
PERdjfAttr.arealsm = (wrk_lsm(:)'*wrk_xyc)';

wrk_xct = countcats(PERid_djf,2);
PERdjfAttr.xt = reshape(atan2(F_sin(:,end)'*wrk_xct(:,:), F_cos(:,end)'*wrk_xct(:,:)),[nCat ds_djf(3)])'/2/pi*ds(1);
PERdjfAttr.xt = 0.5 + mod(PERdjfAttr.xt-0.5, ds(1));  % range from 0.5 - ds(1).5
PERdjfAttr.xt = 0.5 + mod(PERdjfAttr.xt-0.5, ds(1));  % bug
PERdjfAttr.yt = reshape(F_y(1,:)*wrk_ytc(:,:),[ds_djf(3) nCat])./area_tc;
PERdjfAttr.xt(isnan(PERdjfAttr.yt)) = nan;
%PERdjfAttr.ut = %TODO
disp('before loop'); toc

PERdjfAttr.areat = cell([nCat 1]);
for nBlock = 1:nCat
  PERdjfAttr.areat{nBlock} = area_tc(area_tc(:,nBlock)~=0, nBlock);
%  PERdjfAttr.tstart(nBlock) = find(area_tc(:,nBlock),1);
%  PERdjfAttr.tend(nBlock) = find(area_tc(:,nBlock),1,'last');
  PERdjfAttr.tstart(nBlock) = find(any(wrk_ytc(:,:,nBlock),1),1);
  PERdjfAttr.tend(nBlock) = find(any(wrk_ytc(:,:,nBlock),1),1,'last');
end

clear wrk_ytc wrk_xyc wrk_xct F_y F_repmat F_sin F_cos wrk_lsm
clear area_tc
disp('finish attr'); toc
[~,csort] = sort(PERjjaAttr.area,'descend');

else % any PER
PERdjfAttr.id = [];
end % any PER

save(['../index_wise/BlockFreq_',verX,ver,'.mat'],'ver','strTitle','prm','timeNan','lont42','latt42','PERid_jja','PERjjaAttr','PERid_djf','PERdjfAttr','PER_jja','PER_djf','Wgt_jja','Wgt_djf', 'Hot_n','Hotw_n','PERjja_n','PERwjja_n','PERljja_n','PERlwjja_n','-v7.3');
%disp('pause');pause
% 'time','PER','Wgt',

%% associate things
%
Multid_jja = removecats(Hotid_jja.*PERid_jja);
ids = reshape(str2double(split(categories(Multid_jja))),[],2);
%ids = (split(categories(Multid_jja)));
numPER_Hot = countcats(categorical(ids(:,1),str2double(HotAttr.id)),1);
%summary(categorical(numPER_Hot))
areaHot_PER = zeros([numel(PERjjaAttr.id) 1],'single');
for m = 1:size(ids,1)
  areaHot_PER(ids(m,2)) = areaHot_PER(ids(m,2))+HotAttr.area(ids(m,1))/numPER_Hot(ids(m,1));
end
%sum(HotAttr.area(numPER_Hot~=0))/sum(HotAttr.area)

Multid_djf = removecats(Coldid_djf.*circshift(PERid_djf,[11 -5 0]));  %TODO
ids = reshape(str2double(split(categories(Multid_djf))),[],2);
%ids = (split(categories(Multid_djf)));
numPER_Cold = countcats(categorical(ids(:,1),str2double(ColdAttr.id)),1);
summary(categorical(numPER_Cold))
areaCold_PER = zeros([numel(PERdjfAttr.id) 1],'single');
for m = 1:size(ids,1)
  areaCold_PER(ids(m,2)) = areaCold_PER(ids(m,2))+ColdAttr.area(ids(m,1))/numPER_Cold(ids(m,1));
end
%sum(ColdAttr.area(numPER_Cold~=0))/sum(ColdAttr.area)



%disp('pause');pause
%% TODO plot
addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added
rng(2017);

fn_figure = ['../index_wise/all',verX,ver,'.ps'];
system(['rm ',fn_figure]);

%% scatter
%
xdata=PERjjaAttr.area; ydata=areaHot_PER;
hotstat = [polyfit(xdata,ydata,1) corr(xdata(:),ydata(:))];
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(2,3,1)
hold on;
  xlabel({'Blocking area (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
%plot(xdata,ydata,'.','markersize',3);
%plot([min(xdata),max(xdata)],[min(xdata),max(xdata)]*hotstat(1)+hotstat(2),'-','linewidth',2)

%  f_cm = @(nBlock) hsv2rgb([mod(PERjjaAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
%  cm=parula(100); f_cm = @(nBlock) cm(round(1+(interp1(1:ds(2),latt42,PERjjaAttr.y(nBlock))-30)/(90-30)*99 ),:);
  cm=parula(100); f_cm = @(nBlock) cm(round(interp1([0 47 60 ds(2)+0.1],[1 1 100 100],PERjjaAttr.y(nBlock))),:);
  colormap(gca,cm); caxis(latt42([47 60])); colorbar;
  for nBlock = 1:numel(PERjjaAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERjjaAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
%    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
%title({strTitle, ['hot, frac=',num2str(sum(HotAttr.area(numPER_Hot~=0))/sum(HotAttr.area),'%.3f'),', r=',num2str(hotstat(3),'%+.3f')],'color: lat'}, 'interpreter','none');
title({'color: lat'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,2)
hold on;
  xlabel({'Blocking area (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  f_cm = @(nBlock) hsv2rgb([mod(PERjjaAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
  colormap(gca,hsv); caxis([0 360]); colorbar;
  for nBlock = 1:numel(PERjjaAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERjjaAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
%title({'color: lon'}, 'interpreter','none');
title({strTitle, ['hot, frac=',num2str(sum(HotAttr.area(numPER_Hot~=0))/sum(HotAttr.area),'%.3f'),', r=',num2str(hotstat(3),'%+.3f')],'color: lon'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,3)
hold on;
  xlabel({'Blocking area (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  cm=parula(max(PERjjaAttr.tend-PERjjaAttr.tstart+1)); f_cm = @(nBlock) cm(PERjjaAttr.tend(nBlock)-PERjjaAttr.tstart(nBlock)+1,:);
  colormap(gca,cm); caxis([0.5 max(PERjjaAttr.tend-PERjjaAttr.tstart)+1.5]); colorbar;
  for nBlock = 1:numel(PERjjaAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERjjaAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
title({'color: t length'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

wrkid = PERid_jja; wrkid(repmat(~lsm_jja,[1 1 ds_jja(3)]))='<undefined>';
wrk_ycn = countcats(reshape(permute(wrkid,[2 1 3]),ds(2),ds(1)*nd_jja,nyr),2);
PERlg_n = areaEarth/ds(1)/nd_jja*squeeze(sum(wrk_ycn(:,areaHot_PER>0,:),2));
PERlb_n = areaEarth/ds(1)/nd_jja*squeeze(sum(wrk_ycn(:,areaHot_PER==0,:),2));
wrk_ycn = countcats(reshape(permute(PERid_jja,[2 1 3]),ds(2),ds(1)*nd_jja,nyr),2);
PERg_n = areaEarth/ds(1)/nd_jja*squeeze(sum(wrk_ycn(:,areaHot_PER>0,:),2));
PERb_n = areaEarth/ds(1)/nd_jja*squeeze(sum(wrk_ycn(:,areaHot_PER==0,:),2));
wrk_ycn = countcats(reshape(permute(Hotid_jja,[2 1 3]),ds(2),ds(1)*nd_jja,nyr),2);
Hotg_n = areaEarth/ds(1)/nd_jja*squeeze(sum(wrk_ycn(:,numPER_Hot>0,:),2));
Hotb_n = areaEarth/ds(1)/nd_jja*squeeze(sum(wrk_ycn(:,numPER_Hot==0,:),2));
%wrk_yt = squeeze(mean(PER_jja,1));
%bjjaArea_t = areaEarth*wrk_yt(:,:);
%mean(reshape(bjjaArea_t,nd_jja,nyr),1) - PERg_n - PERb_n
%disp(sprintf('jja: %+.3f %+.3f',corr(PERg_n(:)+PERb_n(:),Hotg_n(:)+Hotb_n(:)), corr(PERlg_n(:)+PERlb_n(:),Hotg_n(:)+Hotb_n(:))));

subplot(2,3,6)
hold on;
  xlabel({'Matched blocking area (km^2)'});
  ylabel({'Matched extreme area (km^2)'});
  xdata=PERg_n; ydata=Hotg_n;
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
  for yyyy = yStart:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

%print(gcf,'-dpdf',['scatter_',verX,ver,'.pdf']);
print(gcf, '-dpsc2','-append',fn_figure);
%

%% scatter on land
%
xdata=PERjjaAttr.arealsm; ydata=areaHot_PER;
hotstat = [polyfit(xdata,ydata,1) corr(xdata(:),ydata(:))];
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(2,3,1)
hold on;
  xlabel({'Blocking area on land (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
%plot(xdata,ydata,'.','markersize',3);
%plot([min(xdata),max(xdata)],[min(xdata),max(xdata)]*hotstat(1)+hotstat(2),'-','linewidth',2)

%  f_cm = @(nBlock) hsv2rgb([mod(PERjjaAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
%  cm=parula(100); f_cm = @(nBlock) cm(round(1+(interp1(1:ds(2),latt42,PERjjaAttr.y(nBlock))-30)/(90-30)*99 ),:);
  cm=parula(100); f_cm = @(nBlock) cm(round(interp1([0 47 60 ds(2)],[1 1 100 100],PERjjaAttr.y(nBlock))),:);
  colormap(gca,cm); caxis(latt42([47 60])); colorbar;
  for nBlock = 1:numel(PERjjaAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERjjaAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
%    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
%title({strTitle, ['hot, frac=',num2str(sum(HotAttr.area(numPER_Hot~=0))/sum(HotAttr.area),'%.3f'),', r=',num2str(hotstat(3),'%+.3f')],'color: lat'}, 'interpreter','none');
title({'color: lat'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,2)
hold on;
  xlabel({'Blocking area on land (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  f_cm = @(nBlock) hsv2rgb([mod(PERjjaAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
  colormap(gca,hsv); caxis([0 360]); colorbar;
  for nBlock = 1:numel(PERjjaAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERjjaAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
%title({'color: lon'}, 'interpreter','none');
title({strTitle, ['hot, frac=',num2str(sum(HotAttr.area(numPER_Hot~=0))/sum(HotAttr.area),'%.3f'),', r=',num2str(hotstat(3),'%+.3f')],'color: lon'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,3)
hold on;
  xlabel({'Blocking area on land (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  cm=parula(max(PERjjaAttr.tend-PERjjaAttr.tstart+1)); f_cm = @(nBlock) cm(PERjjaAttr.tend(nBlock)-PERjjaAttr.tstart(nBlock)+1,:);
  colormap(gca,cm); caxis([0.5 max(PERjjaAttr.tend-PERjjaAttr.tstart)+1.5]); colorbar;
  for nBlock = 1:numel(PERjjaAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERjjaAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
title({'color: t length'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,4)
hold on;
  xlabel({'Blocking area (km^2)'});
  ylabel({'Extreme area (km^2)'});
  xdata=PERg_n+PERb_n; ydata=Hotg_n+Hotb_n;
%plot(xdata,ydata,'.','markersize',3);
%plot([min(xdata),max(xdata)],[min(xdata),max(xdata)]*hotstat(1)+hotstat(2),'-','linewidth',2)
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);

  for yyyy = yStart:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
%    text(double(xdata(yyyy)),double(ydata(yyyy)), PERjjaAttr.id{yyyy},'color',f_cm(yyyy),'HorizontalAlignment','center' );
    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,5)
hold on;
  xlabel({'Blocking area on land (km^2)'});
  ylabel({'Extreme area (km^2)'});
  xdata=PERlg_n+PERlb_n; ydata=Hotg_n+Hotb_n;
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
  for yyyy = yStart:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,6)
hold on;
  xlabel({'Matched blocking area on land (km^2)'});
  ylabel({'Matched extreme area (km^2)'});
  xdata=PERlg_n; ydata=Hotg_n;
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
  for yyyy = yStart:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

%print(gcf,'-dpdf',['scatter_',verX,ver,'.pdf']);
print(gcf, '-dpsc2','-append',fn_figure);
%

%{
%if (contains(textTH, 'chunk'))
  disp(['hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',ydata))]);
  system(['echo hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',ydata)), ' >> corrtrend-hot']);
  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',xdata))]);
  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',xdata)), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(xdata,ydata);
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(3:end),Yedges,Ncounts(3:end,:).^0.50','k');  % TODO
  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.50',3,'k');
end
%}

%% Block Freq (Pfahl2a in xtrm_colocate_pchan)
% org
%
PERfreq_jja = mean(PER_jja,3);
PERfreq_djf = mean(PER_djf,3);
wrk_xyc = countcats(PERid_jja,3);
PERgfreq_jja = sum(wrk_xyc(:,:,areaHot_PER>0),3)/ds_jja(3);
PERbfreq_jja = sum(wrk_xyc(:,:,areaHot_PER==0),3)/ds_jja(3);
wrk_xyc = countcats(Hotid_jja,3);
Hotgfreq_jja = sum(wrk_xyc(:,:,numPER_Hot>0),3)/ds_jja(3);
Hotbfreq_jja = sum(wrk_xyc(:,:,numPER_Hot==0),3)/ds_jja(3);

%PERfreq_jja(PERfreq_jja==0) = nan;
%PERfreq_djf(PERfreq_djf==0) = nan;
PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,3,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
%colormap(gca,b2r(0,8)); caxis([0 8]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k')
title({'JJA blocking frequency (%)'},'fontsize',16);
tightmap;

subplot(2,3,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERgfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
%colormap(gca,b2r(0,8)); caxis([0 8]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k')
title({strTitle,'matched blocking'},'fontsize',16);
tightmap;

subplot(2,3,3);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERbfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
%colormap(gca,b2r(0,8)); caxis([0 8]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k')
title({'unmatched blocking'},'fontsize',16);
tightmap;

subplot(2,3,4);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*(Hotgfreq_jja+Hotbfreq_jja)'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
%colormap(gca,b2r(0,1)); caxis([0 1]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k')
title({'JJA extreme frequency (%)'},'fontsize',16);
tightmap;

subplot(2,3,5);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*Hotgfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
%colormap(gca,b2r(0,1)); caxis([0 1]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k')
title({strTitleX,'matched extreme'},'fontsize',16);
tightmap;

subplot(2,3,6);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*Hotbfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
%colormap(gca,b2r(0,1)); caxis([0 1]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k')
title({'unmatched extreme'},'fontsize',16);
tightmap;

clear wrk_xyc
print(gcf, '-dpsc2','-append',fn_figure);
%

%% Block freq trend
wrk_xyc = countcats(PERid_jja,3)/nd_jja;
Weight_c = (PERjjaAttr.yr - mean(yStart:yEnd)) /sumsqr((yStart:yEnd) -mean(yStart:yEnd));
PERgjja_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(areaHot_PER>0),[1 1 numel(PERjjaAttr.id)]),[ds(1:2) 1]),3);
PERbjja_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(areaHot_PER==0),[1 1 numel(PERjjaAttr.id)]),[ds(1:2) 1]),3);
PERjja_trend = PERgjja_trend+PERbjja_trend;
wrk_xyc = countcats(Hotid_jja,3)/nd_jja;
Weight_c = (HotAttr.yr - mean(yStart:yEnd)) /sumsqr((yStart:yEnd) -mean(yStart:yEnd));
Hotg_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(numPER_Hot>0),[1 1 numel(HotAttr.id)]),[ds(1:2) 1]),3);
Hotb_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(numPER_Hot==0),[1 1 numel(HotAttr.id)]),[ds(1:2) 1]),3);
Hot_trend = Hotg_trend+Hotb_trend;
%PER_jja(:,1:floor(ds(2)/2),:) = false;
%  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%Weight_t = 1:nyr;
%Weight_t = Weight_t - mean(Weight_t);
%Weight_t = Weight_t / sumsqr(Weight_t);
%max(max(abs( PERjja_trend - sum(PERjja_xyn.*repmat(reshape(Weight_t,[1 1 nyr]),[ds(1:2) 1]),3) )))
%Hot_jja(:,1:floor(ds(2)/2),:) = false;
%  Hot_xyn = squeeze(mean( reshape(Hot_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%max(max(abs( Hot_trend - sum(Hot_xyn.*repmat(reshape(Weight_t,[1 1 nyr]),[ds(1:2) 1]),3) )))

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,3,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*PERjja_trend'); shading flat;
colormap(gca,b2r(-5,5)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['JJA blocking trends (%/30yr)']},'fontsize',16);
tightmap;

subplot(2,3,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*PERgjja_trend'); shading flat;
colormap(gca,b2r(-5,5)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({strTitle,['matched blocking trends']},'fontsize',16);
tightmap;

subplot(2,3,3);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*PERbjja_trend'); shading flat;
colormap(gca,b2r(-5,5)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['unmatched blocking trends']},'fontsize',16);
tightmap;

subplot(2,3,4);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*Hot_trend'); shading flat;
colormap(gca,b2r(-2,2)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['JJA extreme trends (%/30yr)']},'fontsize',16);
tightmap;

subplot(2,3,5);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*Hotg_trend'); shading flat;
colormap(gca,b2r(-2,2)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({strTitle,['matched extreme trends']},'fontsize',16);
tightmap;

subplot(2,3,6);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*Hotb_trend'); shading flat;
colormap(gca,b2r(-2,2)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['unmatched extreme trends']},'fontsize',16);
tightmap;

clear wrk_xyc
print(gcf, '-dpsc2','-append',fn_figure);

%% year time series (aka legend)
%
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(3,2,1);
hold on;
%  xlabel({'year'}); title({'Mean JJA blocking area (km^2)'});
  xlabel({['year, r=',num2str(corr((yStart:yEnd)',PERg_n'+PERb_n'),'%+.2f')]}); title({'JJA blocking (km^2)'});
plot([yStart:yEnd], PERg_n+PERb_n,'k');
plot([yStart:yEnd], PERg_n,'r');
plot([yStart:yEnd], PERb_n,'b');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERg_n+PERb_n,1),x), [yStart yEnd], 'k--');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERg_n,1),x), [yStart yEnd], 'r--');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERb_n,1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 max(PERg_n+PERb_n)]); grid on; %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,2);
hold on;
  xlabel({'year'}); title({'Ratio'});
plot([yStart:yEnd], PERg_n./(PERg_n+PERb_n),'r');
%plot([yStart:yEnd], PERb_n./(PERg_n+PERb_n),'b');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERg_n./(PERg_n+PERb_n),1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERb_n./(PERg_n+PERb_n),1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 1]); grid on; %axis square;
set(gca,'fontsize',20);

subplot(3,2,3);
hold on;
%  xlabel({'year'}); title({'Mean JJA blocking area on land (km^2)'});
  xlabel({['year, r=',num2str(corr((yStart:yEnd)',PERlg_n'+PERlb_n'),'%+.2f')]}); title({'JJA blocking on land (km^2)'});
plot([yStart:yEnd], PERlg_n+PERlb_n,'k');
plot([yStart:yEnd], PERlg_n,'r');
plot([yStart:yEnd], PERlb_n,'b');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlg_n+PERlb_n,1),x), [yStart yEnd], 'k--');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlg_n,1),x), [yStart yEnd], 'r--');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlb_n,1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 max(PERlg_n+PERlb_n)]); grid on; %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,4);
hold on;
  xlabel({'year'}); title({'Ratio'});
plot([yStart:yEnd], PERlg_n./(PERlg_n+PERlb_n),'r');
%plot([yStart:yEnd], PERlb_n./(PERlg_n+PERlb_n),'b');
    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlg_n./(PERlg_n+PERlb_n),1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlb_n./(PERlg_n+PERlb_n),1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 1]); grid on; %axis square;
set(gca,'fontsize',20);

subplot(3,2,5);
hold on;
%  xlabel({'year'}); title({'Mean JJA extreme area (km^2)'});
  xlabel({['year, r=',num2str(corr((yStart:yEnd)',Hotg_n'+Hotb_n'),'%+.2f')]}); title({'JJA extreme (km^2)'});
plot([yStart:yEnd], Hotg_n+Hotb_n,'k');
plot([yStart:yEnd], Hotg_n,'r');
plot([yStart:yEnd], Hotb_n,'b');
    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotg_n+Hotb_n,1),x), [yStart yEnd], 'k--');
    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotg_n,1),x), [yStart yEnd], 'r--');
    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotb_n,1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 max(Hotg_n+Hotb_n)]); grid on; %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,6);
hold on;
  xlabel({'year'}); title({'Ratio'});
plot([yStart:yEnd], Hotg_n./(Hotg_n+Hotb_n),'r');
%plot([yStart:yEnd], Hotb_n./(Hotg_n+Hotb_n),'b');
    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotg_n./(Hotg_n+Hotb_n),1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotb_n./(Hotg_n+Hotb_n),1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 1]); grid on; %axis square;
set(gca,'fontsize',20);

clear wrk_ycn
print(gcf, '-dpsc2','-append',fn_figure);

%% SeasonalCycle, Area in xtrm_reanalysis_pchan02.m
%
%disp('pause');pause
%wrkid = PERid_jja; wrkid(repmat(~lsm_jja,[1 1 ds_jja(3)]))='<undefined>';
wrk_ycd = countcats(reshape(permute(reshape(wrkid,[ds(1:2),nd_jja,nyr]),[2 1 4 3]),ds(2),ds(1)*nyr,nd_jja),2);
PERlg_d = areaEarth/ds(1)/nyr*squeeze(sum(wrk_ycd(:,areaHot_PER>0,:),2));
PERlb_d = areaEarth/ds(1)/nyr*squeeze(sum(wrk_ycd(:,areaHot_PER==0,:),2));
wrk_ycd = countcats(reshape(permute(reshape(PERid_jja,[ds(1:2),nd_jja,nyr]),[2 1 4 3]),ds(2),ds(1)*nyr,nd_jja),2);
PERg_d = areaEarth/ds(1)/nyr*squeeze(sum(wrk_ycd(:,areaHot_PER>0,:),2));
PERb_d = areaEarth/ds(1)/nyr*squeeze(sum(wrk_ycd(:,areaHot_PER==0,:),2));
wrk_ycd = countcats(reshape(permute(reshape(Hotid_jja,[ds(1:2),nd_jja,nyr]),[2 1 4 3]),ds(2),ds(1)*nyr,nd_jja),2);
Hotg_d = areaEarth/ds(1)/nyr*squeeze(sum(wrk_ycd(:,numPER_Hot>0,:),2));
Hotb_d = areaEarth/ds(1)/nyr*squeeze(sum(wrk_ycd(:,numPER_Hot==0,:),2));
%wrk_yt = squeeze(mean(PER_jja,1));
%bjjaArea_t = areaEarth*wrk_yt(:,:);
%mean(reshape(bjjaArea_t,nd_jja,nyr),2)' - PERg_d - PERb_d

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(3,2,1);
hold on;
  xlabel({'day'}); title({'Mean JJA blocking area (km^2)'});
plot([1:nd_jja], PERg_d+PERb_d,'k');
%plot([1:nd_jja], PERg_d+PERb_d,'g');
plot([1:nd_jja], PERg_d,'r');
plot([1:nd_jja], PERb_d,'b');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERg_d+PERb_d,1),x), [yStart yEnd], 'k--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERg_d,1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERb_d,1),x), [yStart yEnd], 'b--');
ylim([0 max(PERg_d+PERb_d)]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,2);
hold on;
  xlabel({'day'}); title({'Ratio'});
plot([1:nd_jja], PERg_d./(PERg_d+PERb_d),'r');
%plot([1:nd_jja], PERb_d./(PERg_d+PERb_d),'b');
ylim([0 1]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,3);
hold on;
  xlabel({'day'}); title({'Mean JJA blocking area on land (km^2)'});
plot([1:nd_jja], PERlg_d+PERlb_d,'k');
%plot([1:nd_jja], PERlg_d+PERlb_d,'g');
plot([1:nd_jja], PERlg_d,'r');
plot([1:nd_jja], PERlb_d,'b');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlg_d+PERlb_d,1),x), [yStart yEnd], 'k--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlg_d,1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),PERlb_d,1),x), [yStart yEnd], 'b--');
ylim([0 max(PERlg_d+PERlb_d)]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,4);
hold on;
  xlabel({'day'}); title({'Ratio'});
plot([1:nd_jja], PERlg_d./(PERlg_d+PERlb_d),'r');
%plot([1:nd_jja], PERlb_d./(PERlg_d+PERlb_d),'b');
ylim([0 1]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,5);
hold on;
  xlabel({'day'}); title({'Mean JJA extreme area (km^2)'});
plot([1:nd_jja], Hotg_d+Hotb_d,'k');
plot([1:nd_jja], Hotg_d,'r');
plot([1:nd_jja], Hotb_d,'b');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotg_d+Hotb_d,1),x), [yStart yEnd], 'k--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotg_d,1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart:yEnd),Hotb_d,1),x), [yStart yEnd], 'b--');
ylim([0 max(Hotg_d+Hotb_d)]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,6);
hold on;
  xlabel({'day'}); title({'Ratio'});
plot([1:nd_jja], Hotg_d./(Hotg_d+Hotb_d),'r');
%plot([1:nd_jja], Hotb_d./(Hotg_d+Hotb_d),'b');
ylim([0 1]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

clear wrk_ycd
print(gcf, '-dpsc2','-append',fn_figure);
%

%% scatter
%
if (any(PER_djf(:)))
xdata=PERdjfAttr.area; ydata=areaCold_PER;
coldstat = [polyfit(xdata,ydata,1) corr(xdata(:),ydata(:))];
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(2,3,1)
hold on;
  xlabel({'Blocking area (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
%plot(xdata,ydata,'.','markersize',3);
%plot([min(xdata),max(xdata)],[min(xdata),max(xdata)]*coldstat(1)+coldstat(2),'-','linewidth',2)

%  f_cm = @(nBlock) hsv2rgb([mod(PERdjfAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
%  cm=parula(100); f_cm = @(nBlock) cm(round(1+(interp1(1:ds(2),latt42,PERdjfAttr.y(nBlock))-30)/(90-30)*99 ),:);
  cm=parula(100); f_cm = @(nBlock) cm(round(interp1([0 47 60 ds(2)+0.1],[1 1 100 100],PERdjfAttr.y(nBlock))),:);
  colormap(gca,cm); caxis(latt42([47 60])); colorbar;
  for nBlock = 1:numel(PERdjfAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERdjfAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
%    text(double(xdata(yyyy-yStart)),double(ydata(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
%title({strTitle, ['cold, frac=',num2str(sum(ColdAttr.area(numPER_Cold~=0))/sum(ColdAttr.area),'%.3f'),', r=',num2str(coldstat(3),'%+.3f')],'color: lat'}, 'interpreter','none');
title({'color: lat'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,2)
hold on;
  xlabel({'Blocking area (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  f_cm = @(nBlock) hsv2rgb([mod(PERdjfAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
  colormap(gca,hsv); caxis([0 360]); colorbar;
  for nBlock = 1:numel(PERdjfAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERdjfAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
%title({'color: lon'}, 'interpreter','none');
title({strTitle, ['cold, frac=',num2str(sum(ColdAttr.area(numPER_Cold~=0))/sum(ColdAttr.area),'%.3f'),', r=',num2str(coldstat(3),'%+.3f')],'color: lon'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,3)
hold on;
  xlabel({'Blocking area (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  cm=parula(max(PERdjfAttr.tend-PERdjfAttr.tstart+1)); f_cm = @(nBlock) cm(PERdjfAttr.tend(nBlock)-PERdjfAttr.tstart(nBlock)+1,:);
  colormap(gca,cm); caxis([0.5 max(PERdjfAttr.tend-PERdjfAttr.tstart)+1.5]); colorbar;
  for nBlock = 1:numel(PERdjfAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERdjfAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
title({'color: t length'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

wrkid = circshift(PERid_djf,[11 -5 0]); wrkid(repmat(~lsm_djf,[1 1 ds_djf(3)]))='<undefined>';
wrk_ycn = countcats(reshape(permute(wrkid,[2 1 3]),ds(2),ds(1)*nd_djf,nyr-1),2);
PERlg_n = areaEarth/ds(1)/nd_djf*squeeze(sum(wrk_ycn(:,areaCold_PER>0,:),2));
PERlb_n = areaEarth/ds(1)/nd_djf*squeeze(sum(wrk_ycn(:,areaCold_PER==0,:),2));
wrk_ycn = countcats(reshape(permute(PERid_djf,[2 1 3]),ds(2),ds(1)*nd_djf,nyr-1),2);
PERg_n = areaEarth/ds(1)/nd_djf*squeeze(sum(wrk_ycn(:,areaCold_PER>0,:),2));
PERb_n = areaEarth/ds(1)/nd_djf*squeeze(sum(wrk_ycn(:,areaCold_PER==0,:),2));
wrk_ycn = countcats(reshape(permute(Coldid_djf,[2 1 3]),ds(2),ds(1)*nd_djf,nyr-1),2);
Coldg_n = areaEarth/ds(1)/nd_djf*squeeze(sum(wrk_ycn(:,numPER_Cold>0,:),2));
Coldb_n = areaEarth/ds(1)/nd_djf*squeeze(sum(wrk_ycn(:,numPER_Cold==0,:),2));
%wrk_yt = squeeze(mean(PER_djf,1));
%bdjfArea_t = areaEarth*wrk_yt(:,:);
%mean(reshape(bdjfArea_t,nd_djf,nyr-1),1) - PERg_n - PERb_n
disp(sprintf('djf: %+.3f %+.3f',corr(PERg_n(:)+PERb_n(:),Coldg_n(:)+Coldb_n(:)), corr(PERlg_n(:)+PERlb_n(:),Coldg_n(:)+Coldb_n(:))));

subplot(2,3,6)
hold on;
  xlabel({'Matched blocking area (km^2)'});
  ylabel({'Matched extreme area (km^2)'});
  xdata=PERg_n; ydata=Coldg_n;
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
  for yyyy = yStart+1:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
    text(double(xdata(yyyy-yStart)),double(ydata(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

%print(gcf,'-dpdf',['scatter_',verX,ver,'.pdf']);
print(gcf, '-dpsc2','-append',fn_figure);
%

%% scatter on land
%
xdata=PERdjfAttr.arealsm; ydata=areaCold_PER;
coldstat = [polyfit(xdata,ydata,1) corr(xdata(:),ydata(:))];
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(2,3,1)
hold on;
  xlabel({'Blocking area on land (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
%plot(xdata,ydata,'.','markersize',3);
%plot([min(xdata),max(xdata)],[min(xdata),max(xdata)]*coldstat(1)+coldstat(2),'-','linewidth',2)

%  f_cm = @(nBlock) hsv2rgb([mod(PERdjfAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
%  cm=parula(100); f_cm = @(nBlock) cm(round(1+(interp1(1:ds(2),latt42,PERdjfAttr.y(nBlock))-30)/(90-30)*99 ),:);
  cm=parula(100); f_cm = @(nBlock) cm(round(interp1([0 47 60 ds(2)+0.1],[1 1 100 100],PERdjfAttr.y(nBlock))),:);  % bug
  colormap(gca,cm); caxis(latt42([47 60])); colorbar;
  for nBlock = 1:numel(PERdjfAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERdjfAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
%    text(double(xdata(yyyy-yStart)),double(ydata(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
%title({strTitle, ['cold, frac=',num2str(sum(ColdAttr.area(numPER_Cold~=0))/sum(ColdAttr.area),'%.3f'),', r=',num2str(coldstat(3),'%+.3f')],'color: lat'}, 'interpreter','none');
title({'color: lat'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,2)
hold on;
  xlabel({'Blocking area on land (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  f_cm = @(nBlock) hsv2rgb([mod(PERdjfAttr.x(nBlock)-1,ds(1))/ds(1) 1 1]);
  colormap(gca,hsv); caxis([0 360]); colorbar;
  for nBlock = 1:numel(PERdjfAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERdjfAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
%title({'color: lon'}, 'interpreter','none');
title({strTitle, ['cold, frac=',num2str(sum(ColdAttr.area(numPER_Cold~=0))/sum(ColdAttr.area),'%.3f'),', r=',num2str(coldstat(3),'%+.3f')],'color: lon'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,3)
hold on;
  xlabel({'Blocking area on land (km^2*day)'});
  ylabel({'Extreme area (km^2*day)'});
  cm=parula(max(PERdjfAttr.tend-PERdjfAttr.tstart+1)); f_cm = @(nBlock) cm(PERdjfAttr.tend(nBlock)-PERdjfAttr.tstart(nBlock)+1,:);
  colormap(gca,cm); caxis([0.5 max(PERdjfAttr.tend-PERdjfAttr.tstart)+1.5]); colorbar;
  for nBlock = 1:numel(PERdjfAttr.id)
%    plot(xdata(nBlock),ydata(nBlock) ,'o','color',f_cm(nBlock));
   if (xdata(nBlock))
    text(double(xdata(nBlock)),double(ydata(nBlock)), PERdjfAttr.id{nBlock},'color',f_cm(nBlock),'HorizontalAlignment','center' );
   end
  end
title({'color: t length'}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,4)
hold on;
  xlabel({'Blocking area (km^2)'});
  ylabel({'Extreme area (km^2)'});
  xdata=PERg_n+PERb_n; ydata=Coldg_n+Coldb_n;
%plot(xdata,ydata,'.','markersize',3);
%plot([min(xdata),max(xdata)],[min(xdata),max(xdata)]*coldstat(1)+coldstat(2),'-','linewidth',2)
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);

  for yyyy = yStart+1:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
%    text(double(xdata(yyyy)),double(ydata(yyyy)), PERdjfAttr.id{yyyy},'color',f_cm(yyyy),'HorizontalAlignment','center' );
    text(double(xdata(yyyy-yStart)),double(ydata(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,5)
hold on;
  xlabel({'Blocking area on land (km^2)'});
  ylabel({'Extreme area (km^2)'});
  xdata=PERlg_n+PERlb_n; ydata=Coldg_n+Coldb_n;
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
  for yyyy = yStart+1:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
    text(double(xdata(yyyy-yStart)),double(ydata(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,3,6)
hold on;
  xlabel({'Matched blocking area on land (km^2)'});
  ylabel({'Matched extreme area (km^2)'});
  xdata=PERlg_n; ydata=Coldg_n;
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
  for yyyy = yStart+1:yEnd
%    plot(xdata(yyyy),ydata(yyyy) ,'o','color',f_cm(yyyy));
    text(double(xdata(yyyy-yStart)),double(ydata(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
title({['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

%print(gcf,'-dpdf',['scatter_',verX,ver,'.pdf']);
print(gcf, '-dpsc2','-append',fn_figure);
%

%{
%if (contains(textTH, 'chunk'))
  disp(['cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',ydata))]);
  system(['echo cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',ydata)), ' >> corrtrend-cold']);
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',xdata))]);
  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',xdata)), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(xdata,ydata);
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(3:end),Yedges,Ncounts(3:end,:).^0.50','k');  % TODO
  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.50',3,'k');
end
%}

%% Block Freq (Pfahl2a in xtrm_colocate_pchan)
% org
%
%PERfreq_jja = mean(PER_jja,3);
PERfreq_djf = mean(PER_djf,3);
wrk_xyc = countcats(PERid_djf,3);
PERgfreq_djf = sum(wrk_xyc(:,:,areaCold_PER>0),3)/ds_djf(3);
PERbfreq_djf = sum(wrk_xyc(:,:,areaCold_PER==0),3)/ds_djf(3);
wrk_xyc = countcats(Coldid_djf,3);
Coldgfreq_djf = sum(wrk_xyc(:,:,numPER_Cold>0),3)/ds_djf(3);
Coldbfreq_djf = sum(wrk_xyc(:,:,numPER_Cold==0),3)/ds_djf(3);

%PERfreq_jja(PERfreq_jja==0) = nan;
%PERfreq_djf(PERfreq_djf==0) = nan;
%PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,3,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({'DJF blocking frequency (%)'},'fontsize',16);
tightmap;

subplot(2,3,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERgfreq_djf'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitle,'matched blocking'},'fontsize',16);
tightmap;

subplot(2,3,3);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERbfreq_djf'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({'unmatched blocking'},'fontsize',16);
tightmap;

subplot(2,3,4);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*(Coldgfreq_djf+Coldbfreq_djf)'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({'DJF extreme frequency (%)'},'fontsize',16);
tightmap;

subplot(2,3,5);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*Coldgfreq_djf'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitleX,'matched extreme'},'fontsize',16);
tightmap;

subplot(2,3,6);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*Coldbfreq_djf'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({'unmatched extreme'},'fontsize',16);
tightmap;

clear wrk_xyc
print(gcf, '-dpsc2','-append',fn_figure);
%

%% Block freq trend
wrk_xyc = countcats(PERid_djf,3)/nd_djf;
Weight_c = (PERdjfAttr.yr - mean(yStart+1:yEnd)) /sumsqr((yStart+1:yEnd) -mean(yStart+1:yEnd));
PERgdjf_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(areaCold_PER>0),[1 1 numel(PERdjfAttr.id)]),[ds(1:2) 1]),3);
PERbdjf_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(areaCold_PER==0),[1 1 numel(PERdjfAttr.id)]),[ds(1:2) 1]),3);
PERdjf_trend = PERgdjf_trend+PERbdjf_trend;
wrk_xyc = countcats(Coldid_djf,3)/nd_djf;
Weight_c = (ColdAttr.yr - mean(yStart+1:yEnd)) /sumsqr((yStart+1:yEnd) -mean(yStart+1:yEnd));
Coldg_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(numPER_Cold>0),[1 1 numel(ColdAttr.id)]),[ds(1:2) 1]),3);
Coldb_trend = sum(wrk_xyc.*repmat(reshape(Weight_c.*(numPER_Cold==0),[1 1 numel(ColdAttr.id)]),[ds(1:2) 1]),3);
Cold_trend = Coldg_trend+Coldb_trend;
%PER_djf(:,1:floor(ds(2)/2),:) = false;
%  PERdjf_xyn = squeeze(mean( reshape(PER_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%Weight_t = 1:nyr-1;
%Weight_t = Weight_t - mean(Weight_t);
%Weight_t = Weight_t / sumsqr(Weight_t);
%max(max(abs( PERdjf_trend - sum(PERdjf_xyn.*repmat(reshape(Weight_t,[1 1 nyr-1]),[ds(1:2) 1]),3) )))
%Cold_djf(:,1:floor(ds(2)/2),:) = false;
%  Cold_xyn = squeeze(mean( reshape(Cold_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%max(max(abs( Cold_trend - sum(Cold_xyn.*repmat(reshape(Weight_t,[1 1 nyr-1]),[ds(1:2) 1]),3) )))

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,3,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*PERdjf_trend'); shading flat;
colormap(gca,b2r(-5,5)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['DJF blocking trends (%/30yr)']},'fontsize',16);
tightmap;

subplot(2,3,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*PERgdjf_trend'); shading flat;
colormap(gca,b2r(-5,5)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({strTitle,['matched blocking trends']},'fontsize',16);
tightmap;

subplot(2,3,3);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*PERbdjf_trend'); shading flat;
colormap(gca,b2r(-5,5)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['unmatched blocking trends']},'fontsize',16);
tightmap;

subplot(2,3,4);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*Cold_trend'); shading flat;
colormap(gca,b2r(-2,2)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['DJF extreme trends (%/30yr)']},'fontsize',16);
tightmap;

subplot(2,3,5);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*Coldg_trend'); shading flat;
colormap(gca,b2r(-2,2)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({strTitle,['matched extreme trends']},'fontsize',16);
tightmap;

subplot(2,3,6);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 30*100*Coldb_trend'); shading flat;
colormap(gca,b2r(-2,2)); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k')
title({['unmatched extreme trends']},'fontsize',16);
tightmap;

clear wrk_xyc
print(gcf, '-dpsc2','-append',fn_figure);

%% year time series (aka legend)
%
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(3,2,1);
hold on;
%  xlabel({'year'}); title({'Mean DJF blocking area (km^2)'});
  xlabel({['year, r=',num2str(corr((yStart+1:yEnd)',PERg_n'+PERb_n'),'%+.2f')]}); title({'DJF blocking (km^2)'});
plot([yStart+1:yEnd], PERg_n+PERb_n,'k');
plot([yStart+1:yEnd], PERg_n,'r');
plot([yStart+1:yEnd], PERb_n,'b');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERg_n+PERb_n,1),x), [yStart yEnd], 'k--');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERg_n,1),x), [yStart yEnd], 'r--');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERb_n,1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 max(PERg_n+PERb_n)]); grid on; %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,2);
hold on;
  xlabel({'year'}); title({'Ratio'});
plot([yStart+1:yEnd], PERg_n./(PERg_n+PERb_n),'r');
%plot([yStart+1:yEnd], PERb_n./(PERg_n+PERb_n),'b');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERg_n./(PERg_n+PERb_n),1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERb_n./(PERg_n+PERb_n),1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 1]); grid on; %axis square;
set(gca,'fontsize',20);

subplot(3,2,3);
hold on;
%  xlabel({'year'}); title({'Mean DJF blocking area on land (km^2)'});
  xlabel({['year, r=',num2str(corr((yStart+1:yEnd)',PERlg_n'+PERlb_n'),'%+.2f')]}); title({'DJF blocking on land (km^2)'});
plot([yStart+1:yEnd], PERlg_n+PERlb_n,'k');
plot([yStart+1:yEnd], PERlg_n,'r');
plot([yStart+1:yEnd], PERlb_n,'b');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlg_n+PERlb_n,1),x), [yStart yEnd], 'k--');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlg_n,1),x), [yStart yEnd], 'r--');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlb_n,1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 max(PERlg_n+PERlb_n)]); grid on; %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,4);
hold on;
  xlabel({'year'}); title({'Ratio'});
plot([yStart+1:yEnd], PERlg_n./(PERlg_n+PERlb_n),'r');
%plot([yStart+1:yEnd], PERlb_n./(PERlg_n+PERlb_n),'b');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlg_n./(PERlg_n+PERlb_n),1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlb_n./(PERlg_n+PERlb_n),1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 1]); grid on; %axis square;
set(gca,'fontsize',20);

subplot(3,2,5);
hold on;
%  xlabel({'year'}); title({'Mean DJF extreme area (km^2)'});
  xlabel({['year, r=',num2str(corr((yStart+1:yEnd)',Coldg_n'+Coldb_n'),'%+.2f')]}); title({'DJF extreme (km^2)'});
plot([yStart+1:yEnd], Coldg_n+Coldb_n,'k');
plot([yStart+1:yEnd], Coldg_n,'r');
plot([yStart+1:yEnd], Coldb_n,'b');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldg_n+Coldb_n,1),x), [yStart yEnd], 'k--');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldg_n,1),x), [yStart yEnd], 'r--');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldb_n,1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 max(Coldg_n+Coldb_n)]); grid on; %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,6);
hold on;
  xlabel({'year'}); title({'Ratio'});
plot([yStart+1:yEnd], Coldg_n./(Coldg_n+Coldb_n),'r');
%plot([yStart+1:yEnd], Coldb_n./(Coldg_n+Coldb_n),'b');
    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldg_n./(Coldg_n+Coldb_n),1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldb_n./(Coldg_n+Coldb_n),1),x), [yStart yEnd], 'b--');
xlim([yStart yEnd]); ylim([0 1]); grid on; %axis square;
set(gca,'fontsize',20);

clear wrk_ycn
print(gcf, '-dpsc2','-append',fn_figure);

%% SeasonalCycle, Area in xtrm_reanalysis_pchan02.m
%
%disp('pause');pause
%wrkid = PERid_djf; wrkid(repmat(~lsm_djf,[1 1 ds_djf(3)]))='<undefined>';
wrk_ycd = countcats(reshape(permute(reshape(wrkid,[ds(1:2),nd_djf,nyr-1]),[2 1 4 3]),ds(2),ds(1)*(nyr-1),nd_djf),2);
PERlg_d = areaEarth/ds(1)/(nyr-1)*squeeze(sum(wrk_ycd(:,areaCold_PER>0,:),2));
PERlb_d = areaEarth/ds(1)/(nyr-1)*squeeze(sum(wrk_ycd(:,areaCold_PER==0,:),2));
wrk_ycd = countcats(reshape(permute(reshape(PERid_djf,[ds(1:2),nd_djf,nyr-1]),[2 1 4 3]),ds(2),ds(1)*(nyr-1),nd_djf),2);
PERg_d = areaEarth/ds(1)/(nyr-1)*squeeze(sum(wrk_ycd(:,areaCold_PER>0,:),2));
PERb_d = areaEarth/ds(1)/(nyr-1)*squeeze(sum(wrk_ycd(:,areaCold_PER==0,:),2));
wrk_ycd = countcats(reshape(permute(reshape(Coldid_djf,[ds(1:2),nd_djf,nyr-1]),[2 1 4 3]),ds(2),ds(1)*(nyr-1),nd_djf),2);
Coldg_d = areaEarth/ds(1)/(nyr-1)*squeeze(sum(wrk_ycd(:,numPER_Cold>0,:),2));
Coldb_d = areaEarth/ds(1)/(nyr-1)*squeeze(sum(wrk_ycd(:,numPER_Cold==0,:),2));
%wrk_yt = squeeze(mean(PER_djf,1));
%bdjfArea_t = areaEarth*wrk_yt(:,:);
%mean(reshape(bdjfArea_t,nd_djf,nyr-1),2)' - PERg_d - PERb_d

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(3,2,1);
hold on;
  xlabel({'day'}); title({'Mean DJF blocking area (km^2)'});
plot([1:nd_djf], PERg_d+PERb_d,'k');
%plot([1:nd_djf], PERg_d+PERb_d,'g');
plot([1:nd_djf], PERg_d,'r');
plot([1:nd_djf], PERb_d,'b');
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERg_d+PERb_d,1),x), [yStart yEnd], 'k--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERg_d,1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERb_d,1),x), [yStart yEnd], 'b--');
ylim([0 max(PERg_d+PERb_d)]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,2);
hold on;
  xlabel({'day'}); title({'Ratio'});
plot([1:nd_djf], PERg_d./(PERg_d+PERb_d),'r');
%plot([1:nd_djf], PERb_d./(PERg_d+PERb_d),'b');
ylim([0 1]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,3);
hold on;
  xlabel({'day'}); title({'Mean DJF blocking area on land (km^2)'});
plot([1:nd_djf], PERlg_d+PERlb_d,'k');
%plot([1:nd_djf], PERlg_d+PERlb_d,'g');
plot([1:nd_djf], PERlg_d,'r');
plot([1:nd_djf], PERlb_d,'b');
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlg_d+PERlb_d,1),x), [yStart yEnd], 'k--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlg_d,1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),PERlb_d,1),x), [yStart yEnd], 'b--');
ylim([0 max(PERlg_d+PERlb_d)]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,4);
hold on;
  xlabel({'day'}); title({'Ratio'});
plot([1:nd_djf], PERlg_d./(PERlg_d+PERlb_d),'r');
%plot([1:nd_djf], PERlb_d./(PERlg_d+PERlb_d),'b');
ylim([0 1]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,5);
hold on;
  xlabel({'day'}); title({'Mean DJF extreme area (km^2)'});
plot([1:nd_djf], Coldg_d+Coldb_d,'k');
plot([1:nd_djf], Coldg_d,'r');
plot([1:nd_djf], Coldb_d,'b');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldg_d+Coldb_d,1),x), [yStart yEnd], 'k--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldg_d,1),x), [yStart yEnd], 'r--');
%    fplot(@(x) polyval(polyfit((yStart+1:yEnd),Coldb_d,1),x), [yStart yEnd], 'b--');
ylim([0 max(Coldg_d+Coldb_d)]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

subplot(3,2,6);
hold on;
  xlabel({'day'}); title({'Ratio'});
plot([1:nd_djf], Coldg_d./(Coldg_d+Coldb_d),'r');
%plot([1:nd_djf], Coldb_d./(Coldg_d+Coldb_d),'b');
ylim([0 1]); grid on; %xlim([yStart yEnd]); %axis square;
%legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

clear wrk_ycd
print(gcf, '-dpsc2','-append',fn_figure);
end % any PER
%





%{

textTH = [textTH, '_chunkwgt'];

if (contains(textTH, 'wgt'))
  Hot_yt  = squeeze(mean(Hot_jja.*(mx2t_jja),1));
  Cold_yt  = squeeze(mean(Cold_djf.*(mn2t_djf),1));
else
  Hot_yt  = squeeze(mean(Hot_jja,1));
  Cold_yt  = squeeze(mean(Cold_djf,1));
end

%ver=['abcd_',caseid];  %
if (contains(textTH, 'wgt'))
  PERjja_yt  = squeeze(mean(PER_jja.*(Wgt_jja),1));
  PERdjf_yt  = squeeze(mean(PER_djf.*(Wgt_djf),1));
else
  PERjja_yt  = squeeze(mean(PER_jja,1));
  PERdjf_yt  = squeeze(mean(PER_djf,1));
end

bjjaArea_t = ( areaEarth * PERjja_yt(:,:) )';
hotArea_t  = ( areaEarth * Hot_yt(:,:) )';
bdjfArea_t = ( areaEarth * PERdjf_yt(:,:) )';
coldArea_t = ( areaEarth * Cold_yt(:,:) )';

if (contains(textTH, 'chunk'))
  bjjaArea_t = squeeze(mean( reshape(bjjaArea_t(:),[],nyr ), 1));
  hotArea_t = squeeze(mean( reshape(hotArea_t(:),[],nyr ), 1));
  bdjfArea_t = squeeze(mean( reshape(bdjfArea_t(:),[],nyr-1 ), 1));
  coldArea_t = squeeze(mean( reshape(coldArea_t(:),[],nyr-1 ), 1));
end

hotstat = [polyfit(bjjaArea_t(:),hotArea_t(:),1) corr(bjjaArea_t(:),hotArea_t(:))];
coldstat = [polyfit(bdjfArea_t(:),coldArea_t(:),1) corr(bdjfArea_t(:),coldArea_t(:))];
disp([textTH,'_',ver,':  ',num2str(hotstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(hotstat(3)), ' >> corr-hot']);
disp([textTH,'_',ver,':  ',num2str(coldstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(coldstat(3)), ' >> corr-cold']);
%disp([textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)])])
%system(['echo ',textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)]), ' >> corr-hot']);

 fn_save  = ['../index_wise/scatter_',textTH,'_',ver,'.mat'];
 save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');

% no chunk for lagcorr
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH, '_wgt'];

bjjaArea_t = ( areaEarth * PERjja_yt(:,:) )';
hotArea_t  = ( areaEarth * Hot_yt(:,:) )';
bdjfArea_t = ( areaEarth * PERdjf_yt(:,:) )';
coldArea_t = ( areaEarth * Cold_yt(:,:) )';

if (contains(textTH, 'chunk'))
  bjjaArea_t = squeeze(mean( reshape(bjjaArea_t(:),[],nyr ), 1));
  hotArea_t = squeeze(mean( reshape(hotArea_t(:),[],nyr ), 1));
  bdjfArea_t = squeeze(mean( reshape(bdjfArea_t(:),[],nyr-1 ), 1));
  coldArea_t = squeeze(mean( reshape(coldArea_t(:),[],nyr-1 ), 1));
end

hotstat = [polyfit(bjjaArea_t(:),hotArea_t(:),1) corr(bjjaArea_t(:),hotArea_t(:))];
coldstat = [polyfit(bdjfArea_t(:),coldArea_t(:),1) corr(bdjfArea_t(:),coldArea_t(:))];
disp([textTH,'_',ver,':  ',num2str(hotstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(hotstat(3)), ' >> corr-hot']);
disp([textTH,'_',ver,':  ',num2str(coldstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(coldstat(3)), ' >> corr-cold']);
%disp([textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)])])
%system(['echo ',textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)]), ' >> corr-hot']);

 fn_save  = ['../index_wise/scatter_',textTH,'_',ver,'.mat'];
 save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');

% lagcorr, not implemented for 'season'
%{
figure;
hold on;
xlabel('Extreme area lags (days)'); ylabel('Correlation r'); grid on;
[r lags] = xcorr(zscore(bArea_ht(1,wrk(1,:))),zscore(hotArea_ht(1,wrk(1,:))), 30,'coeff');
%[r; lags]
[aaa bbb] = max(r);
%disp([textTH,'_',ver,':  ',num2str([aaa,-lags(bbb)])])
%system(['echo ',textTH,'_',ver,':  ',num2str([aaa,-lags(bbb)]), ' >> corrlag-hot']);
disp([textTH,'_',ver,':  ',num2str([prm_angle, aaa,-lags(bbb)])])
system(['echo ',textTH,'_',ver,':  ',num2str([prm_angle, aaa,-lags(bbb)]), ' >> corrlag-hot']);
plot(-lags,r,'r-x')
%[r lags] = xcorr(zscore(bArea_ht(1,wrk(1,:))),zscore(coldArea_ht(1,wrk(1,:))), 30,'coeff');
%[r; lags]
%plot(-lags,r,'b-x')
title({'V850','no integration'}, 'interpreter','none')
legend({'hot'}); legend('boxoff'); axis([-30 30 -0.05 1])
%legend({'hot','cold'}); legend('boxoff'); axis([-30 30 -0.05 1])
%}

textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH, '_chunk'];

if (contains(textTH, 'wgt'))
  Hot_yt  = squeeze(mean(Hot_jja.*(mx2t_jja),1));
  Cold_yt  = squeeze(mean(Cold_djf.*(mn2t_djf),1));
else
  Hot_yt  = squeeze(mean(Hot_jja,1));
  Cold_yt  = squeeze(mean(Cold_djf,1));
end

%ver=['abcd_',caseid];  %
if (contains(textTH, 'wgt'))
  PERjja_yt  = squeeze(mean(PER_jja.*(Wgt_jja),1));
  PERdjf_yt  = squeeze(mean(PER_djf.*(Wgt_djf),1));
else
  PERjja_yt  = squeeze(mean(PER_jja,1));
  PERdjf_yt  = squeeze(mean(PER_djf,1));
end

bjjaArea_t = ( areaEarth * PERjja_yt(:,:) )';
hotArea_t  = ( areaEarth * Hot_yt(:,:) )';
bdjfArea_t = ( areaEarth * PERdjf_yt(:,:) )';
coldArea_t = ( areaEarth * Cold_yt(:,:) )';

if (contains(textTH, 'chunk'))
  bjjaArea_t = squeeze(mean( reshape(bjjaArea_t(:),[],nyr ), 1));
  hotArea_t = squeeze(mean( reshape(hotArea_t(:),[],nyr ), 1));
  bdjfArea_t = squeeze(mean( reshape(bdjfArea_t(:),[],nyr-1 ), 1));
  coldArea_t = squeeze(mean( reshape(coldArea_t(:),[],nyr-1 ), 1));
end

hotstat = [polyfit(bjjaArea_t(:),hotArea_t(:),1) corr(bjjaArea_t(:),hotArea_t(:))];
coldstat = [polyfit(bdjfArea_t(:),coldArea_t(:),1) corr(bdjfArea_t(:),coldArea_t(:))];
disp([textTH,'_',ver,':  ',num2str(hotstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(hotstat(3)), ' >> corr-hot']);
disp([textTH,'_',ver,':  ',num2str(coldstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(coldstat(3)), ' >> corr-cold']);
%disp([textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)])])
%system(['echo ',textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)]), ' >> corr-hot']);

 fn_save  = ['../index_wise/scatter_',textTH,'_',ver,'.mat'];
 save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');

% no chunk for lagcorr
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH, '_org'];

bjjaArea_t = ( areaEarth * PERjja_yt(:,:) )';
hotArea_t  = ( areaEarth * Hot_yt(:,:) )';
bdjfArea_t = ( areaEarth * PERdjf_yt(:,:) )';
coldArea_t = ( areaEarth * Cold_yt(:,:) )';

if (contains(textTH, 'chunk'))
  bjjaArea_t = squeeze(mean( reshape(bjjaArea_t(:),[],nyr ), 1));
  hotArea_t = squeeze(mean( reshape(hotArea_t(:),[],nyr ), 1));
  bdjfArea_t = squeeze(mean( reshape(bdjfArea_t(:),[],nyr-1 ), 1));
  coldArea_t = squeeze(mean( reshape(coldArea_t(:),[],nyr-1 ), 1));
end

hotstat = [polyfit(bjjaArea_t(:),hotArea_t(:),1) corr(bjjaArea_t(:),hotArea_t(:))];
coldstat = [polyfit(bdjfArea_t(:),coldArea_t(:),1) corr(bdjfArea_t(:),coldArea_t(:))];
disp([textTH,'_',ver,':  ',num2str(hotstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(hotstat(3)), ' >> corr-hot']);
disp([textTH,'_',ver,':  ',num2str(coldstat(3))])
system(['echo ',textTH,'_',ver,':  ',num2str(coldstat(3)), ' >> corr-cold']);
%disp([textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)])])
%system(['echo ',textTH,'_',ver,':  ',num2str([prm_angle hotstat(3)]), ' >> corr-hot']);

 fn_save  = ['../index_wise/scatter_',textTH,'_',ver,'.mat'];
 save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');
%

%% plot polyfit
%

%fn_figure = ['../index_wise/all',ver,'.ps'];

%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%
%load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd')

textWgtArr = {'chunk','chunkwgt','org','wgt'};
for textWgt = textWgtArr

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];
%textTH = [textTH, '_chunk'];
%textTH = [textTH, '_chunkwgt'];

%ver=['abcd_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(1,2,1)
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',hotArea_t(:)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
  xlabel({'Weighted blocking area (unit*km^2)',strXCorr});
  ylabel({'Weighted extreme area (K*km^2)',strYCorr});
else
  xlabel({'Blocking area (km^2)',strXCorr});
  ylabel({'Extreme area (km^2)',strYCorr});
end
plot(bjjaArea_t(:),hotArea_t(:),'.','markersize',3);
plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))],[min(bjjaArea_t(:)),max(bjjaArea_t(:))]*hotstat(1)+hotstat(2),'-','linewidth',2)
% http://www.originlab.com/doc/Origin-Help/LR-Algorithm   confidence ellipse
%inv22 = inv( cov(bjjaArea_t(:),hotArea_t(:)) );  % normalize by N-1
%xmean=mean(bjjaArea_t(:)); ymean=mean(hotArea_t(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(bjjaArea_t(:)),max(bjjaArea_t(:)), min(hotArea_t(:)),max(hotArea_t(:))],'r','linewidth',2,'LevelList',1);
%plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))], ([min(bjjaArea_t(:)),max(bjjaArea_t(:))]-xmean)*tan(atan2(-2*inv22(2),inv22(4)-inv22(1))/2)+ymean,'-','linewidth',2)

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart:yEnd
    text(double(bjjaArea_t(yyyy-yStart+1)),double(hotArea_t(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',hotArea_t(:)))]);
  system(['echo hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',hotArea_t(:))), ' >> corrtrend-hot']);
  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)))]);
  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bjjaArea_t(:),hotArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({strTitle, ['hot, r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bjjaArea_t(:)),max(bjjaArea_t(:)), min(hotArea_t(:)),max(hotArea_t(:))]); axis square; %axis tight;

subplot(1,2,2)
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',coldArea_t(:)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
  xlabel({'Weighted blocking area (unit*km^2)',strXCorr});
  ylabel({'Weighted extreme area (K*km^2)',strYCorr});
else
  xlabel({'Blocking area (km^2)',strXCorr});
  ylabel({'Extreme area (km^2)',strYCorr});
end
plot(bdjfArea_t(:),coldArea_t(:),'.','markersize',3);
plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))],[min(bdjfArea_t(:)),max(bdjfArea_t(:))]*coldstat(1)+coldstat(2),'-','linewidth',2)
%inv22 = inv( cov(bdjfArea_t(:),coldArea_t(:)) );  % normalize by N-1
%xmean=mean(bdjfArea_t(:)); ymean=mean(coldArea_t(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))],'r','linewidth',2,'LevelList',1);

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart+1:yEnd
    text(double(bdjfArea_t(yyyy-yStart)),double(coldArea_t(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',coldArea_t(:)))]);
  system(['echo cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',coldArea_t(:))), ' >> corrtrend-cold']);
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bdjfArea_t(:),coldArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title(['cold, r=',num2str(coldstat(3),'%+.3f')])
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))]); axis square; %axis tight;

print(gcf, '-dpsc2','-append',fn_figure);
end % textWgt loop
%system(['ps2pdf ',fn_figure]);
%

%% SeasonalCycle, Area in xtrm_reanalysis_pchan02.m
%
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%load(['temp_',textTH,'_',text,'.mat'], 'nyr',  'yStart','yEnd')

textWgtArr = {'org','wgt'};
for textWgt = textWgtArr

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];

%ver=['abcd_',caseid];
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
%subplot(2,2,1); hold on;
%ax=gca; ax.ColorOrder = jet(nyr);
%xlabel({'Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
%xlabel({'day'}); ylabel('Extreme area (km^2)');
%plot(reshape( hotArea_t,[],nyr ));
%plot([1 nd_jja],HotQuantile(i,j)*[1 1],'k-');
%title(['JJA']);

%ver=['abcd_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(2,2,1); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
if (contains(textTH, 'wgt'))
  xlabel({'day'}); ylabel({'Weighted blocking area (unit*km^2)'});
else
  xlabel({'day'}); ylabel({'Blocking area (km^2)'});
end
plot(reshape( bjjaArea_t,[],nyr ));
plot(mean(reshape( bjjaArea_t,[],nyr ),2),'k','linewidth',1);
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
title([strTitle,': JJA']);

subplot(2,2,3); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
%ylabel('year');
%plot([1;1]*[yStart:yEnd]);
if (contains(textTH, 'wgt'))
  xlabel({'year'}); ylabel({'Mean weighted blocking area (unit*km^2)'});
else
  xlabel({'year'}); ylabel({'Mean blocking area (km^2)'});
end
plot([1;1]*[yStart:yEnd], [zeros(1,nyr); mean( reshape(bjjaArea_t(:),[],nyr ), 1)]);
tmpstat = [polyfit((yStart:yEnd),mean( reshape(bjjaArea_t(:),[],nyr ), 1),1)];
fplot(@(x) polyval(tmpstat,x), [yStart yEnd], 'k');
title(['Legend, r=',num2str(corr((yStart:yEnd)',mean( reshape(bjjaArea_t(:),[],nyr ), 1)'),'%+.3f')]);
xlim([yStart yEnd]);


%ver=['abcd_',caseid];
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
%subplot(2,2,4); hold on;
%ax=gca; ax.ColorOrder = jet(nyr-1);
%xlabel({'Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
%xlabel({'day'}); ylabel('Extreme area (km^2)');
%plot(reshape( coldArea_t,[],nyr-1 ));
%plot([1 nd_jja],ColdQuantile(i,j)*[1 1],'k-');
%title(['DJF']);

%ver=['abcd_',caseid];
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(2,2,2); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
if (contains(textTH, 'wgt'))
  xlabel({'day'}); ylabel({'Weighted blocking area (unit*km^2)'});
else
  xlabel({'day'}); ylabel({'Blocking area (km^2)'});
end
plot(reshape( bdjfArea_t,[],nyr-1 ));
plot(mean(reshape( bdjfArea_t,[],nyr-1 ),2),'k','linewidth',1);
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
title([strTitle,': DJF']);

subplot(2,2,4); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
%ylabel('year');
%plot([1;1]*[yStart+1:yEnd]);
if (contains(textTH, 'wgt'))
  xlabel({'year'}); ylabel({'Mean weighted blocking area (unit*km^2)'});
else
  xlabel({'year'}); ylabel({'Mean blocking area (km^2)'});
end
plot([1;1]*[yStart+1:yEnd], [zeros(1,nyr-1); mean( reshape(bdjfArea_t(:),[],nyr-1 ), 1)]);
tmpstat = [polyfit((yStart+1:yEnd),mean( reshape(bdjfArea_t(:),[],nyr-1 ), 1),1)];
fplot(@(x) polyval(tmpstat,x), [yStart yEnd], 'k');
title(['Legend, r=',num2str(corr((yStart+1:yEnd)',mean( reshape(bdjfArea_t(:),[],nyr-1 ), 1)'),'%+.3f')]);
xlim([yStart+1 yEnd]);

%savefig(gcf,['SeasonalCycleJJAareas_',textTH,'_',text,'.fig'])
%print(gcf,'-dpdf',['SeasonalCycleJJAareas_',textTH,'_',text,'.pdf'])
print(gcf, '-dpsc2','-append',fn_figure);
end % textWgt loop
%

%% Block Freq (Pfahl2a in xtrm_colocate_pchan)
% org
%
PERfreq_jja = mean(PER_jja,3);
PERfreq_djf = mean(PER_djf,3);
%PERfreq_jja(PERfreq_jja==0) = nan;
%PERfreq_djf(PERfreq_djf==0) = nan;
PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_jja'); shading flat;
colormap(jet(15)); caxis([0 30]); colorbar; caxis auto;  % TODO fig2 in Masato 2013a
plotm(coastlat,coastlon,'k')
%title('\fontsize{20}Relative frequency (%) of intense blocking events during JJA');
title({strTitle,'JJA blocking frequency (%)'},'fontsize',16);
tightmap;
%print(gcf,'-dpdf',['Pfahl2a_',textTH,'_',text,'.pdf'])
%savefig(gcf,['Pfahl2a_',textTH,'_',text,'.fig'])

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(jet(15)); caxis([0 30]); colorbar; caxis auto;  % fig2 in Masato 2013a
plotm(coastlat,coastlon,'k')
%title('\fontsize{20}Relative frequency (%) of intense blocking events during DJF');
title({'DJF blocking frequency (%)'},'fontsize',16);
tightmap;
%pause(5);
%print(gcf,'-dpdf',['Pfahl2c_',textTH,'_',text,'.pdf'])
%savefig(gcf,['Pfahl2c_',textTH,'_',text,'.fig'])
print(gcf, '-dpsc2','-append',fn_figure);
%

% wgt
PERfreq_jja = mean(PER_jja.*(Wgt_jja),3);
PERfreq_djf = mean(PER_djf.*(Wgt_djf),3);
%PERfreq_jja(PERfreq_jja==0) = nan;
%PERfreq_djf(PERfreq_djf==0) = nan;
PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_jja'); shading flat;
colormap(jet(15)); caxis([0 30]*80); colorbar; caxis auto;
plotm(coastlat,coastlon,'k');
title({strTitle,'JJA weighted blocking frequency (unit*%)'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(jet(15)); caxis([0 30]*80); colorbar; caxis auto;
plotm(coastlat,coastlon,'k');
title({'DJF weighted blocking frequency (unit*%)'},'fontsize',16);
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);

%% POD (Pfahl2b in xtrm_colocate_pchan)
%{
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolorm(double(lat1a),double(lon1a),double(100*HotPod5000'));
colormap(jet(10)); caxis([0 100]); colorbar;
plotm(coastlat,coastlon,'k')
title('\fontsize{20}Percentage f of six-hourly warm temperature extremes from ERA-Interim occurring simultaneously with an intense blocking event at the same grid point');
%pause(5);
%print(gcf,'-dpdf',['Pfahl2b_',textTH,'_',text,'.pdf'])
%savefig(gcf,['Pfahl2b_',textTH,'_',text,'.fig'])
print(gcf, '-dpsc2','-append',fn_figure);
%}

%% SelectCompo

%% regress blocking on extreme area (Block/extreme)
textWgtArr = {'chunk','chunkwgt'};
for textWgt = textWgtArr

textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])

if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERdjf_xyn = squeeze(mean( reshape(PER_djf.*(Wgt_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%  Hot_xyn = squeeze(mean( reshape(Hot_jja.*(mx2t_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  Cold_xyn = squeeze(mean( reshape(Cold_djf.*(mn2t_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERdjf_xyn = squeeze(mean( reshape(PER_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%  Hot_xyn = squeeze(mean( reshape(Hot_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  Cold_xyn = squeeze(mean( reshape(Cold_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
end
%PERjja_sn  =  reshape( PERjja_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
%PERdjf_sn  =  reshape( PERdjf_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);
%Hot_sn  =  reshape( Hot_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
%Cold_sn  =  reshape( Cold_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);

% -mean
%PERjja_sn = PERjja_sn -repmat(mean(PERjja_sn,2),[1 nyr]);  % double not needed..
%PERdjf_sn = PERdjf_sn -repmat(mean(PERdjf_sn,2),[1 nyr-1]);  % double not needed..
%Hot_sn = Hot_sn -repmat(mean(Hot_sn,2),[1 nyr]);  % double not needed..
%Cold_sn = Cold_sn -repmat(mean(Cold_sn,2),[1 nyr-1]);  % double not needed..
hotArea_t = hotArea_t - nanmean(hotArea_t);
coldArea_t = coldArea_t - nanmean(coldArea_t);

%PER_yht(:,:,timeNan) = 0;

hotArea_t = hotArea_t / sumsqr(hotArea_t);
coldArea_t = coldArea_t / sumsqr(coldArea_t);

PERjja_reg = sum(PERjja_xyn.*repmat(reshape(hotArea_t,[1 1 nyr]),[ds(1:2) 1]),3);
PERdjf_reg = sum(PERdjf_xyn.*repmat(reshape(coldArea_t,[1 1 nyr-1]),[ds(1:2) 1]),3);
PERjja_reg(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
PERdjf_reg(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERjja_reg'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-8e-5,8e-5)); %colorbar; caxis auto;  %TODO
  title({strTitle,['JJA blocking regressed on extreme hot area'],'unit*%/K/km^2'},'fontsize',16);
else
  colormap(gca,b2r(-6e-6,6e-6)); %colorbar; caxis auto;
  title({strTitle,['JJA blocking regressed on extreme hot area'],'%/km^2'},'fontsize',16);
end
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERdjf_reg'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-5e-5,5e-5)); %colorbar; caxis auto;
  title({['DJF blocking regressed on extreme cold area'],'unit*%/K/km^2'},'fontsize',16);
else
  colormap(gca,b2r(-6e-6,6e-6)); %colorbar; caxis auto;
  title({['DJF blocking regressed on extreme cold area'],'%/km^2'},'fontsize',16);
end
tightmap;


print(gcf, '-dpsc2','-append',fn_figure);

end  % textWgt

%% Block freq trend
textWgtArr = {'chunk','chunkwgt'};
for textWgt = textWgtArr

textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])

if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERdjf_xyn = squeeze(mean( reshape(PER_djf.*(Wgt_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%  Hot_xyn = squeeze(mean( reshape(Hot_jja.*(mx2t_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  Cold_xyn = squeeze(mean( reshape(Cold_djf.*(mn2t_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERdjf_xyn = squeeze(mean( reshape(PER_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%  Hot_xyn = squeeze(mean( reshape(Hot_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  Cold_xyn = squeeze(mean( reshape(Cold_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
end
%PERjja_sn  =  reshape( PERjja_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
%PERdjf_sn  =  reshape( PERdjf_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);
%Hot_sn  =  reshape( Hot_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
%Cold_sn  =  reshape( Cold_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);

% -mean
%PERjja_sn = PERjja_sn -repmat(mean(PERjja_sn,2),[1 nyr]);  % double not needed..
%PERdjf_sn = PERdjf_sn -repmat(mean(PERdjf_sn,2),[1 nyr-1]);  % double not needed..
%Hot_sn = Hot_sn -repmat(mean(Hot_sn,2),[1 nyr]);  % double not needed..
%Cold_sn = Cold_sn -repmat(mean(Cold_sn,2),[1 nyr-1]);  % double not needed..

Weight_t = 1:nyr;
Weight_t = Weight_t - mean(Weight_t);
Weight_t = Weight_t / sumsqr(Weight_t);
PERjja_trend = sum(PERjja_xyn.*repmat(reshape(Weight_t,[1 1 nyr]),[ds(1:2) 1]),3);

Weight_t = 1:nyr-1;
Weight_t = Weight_t - mean(Weight_t);
Weight_t = Weight_t / sumsqr(Weight_t);
PERdjf_trend = sum(PERdjf_xyn.*repmat(reshape(Weight_t,[1 1 nyr-1]),[ds(1:2) 1]),3);

PERjja_trend(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
PERdjf_trend(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERjja_trend'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-40,40)); colorbar; %caxis auto;  %TODO
  title({strTitle,['JJA weighted blocking frequency trends'],'unit*%/yr'},'fontsize',16);
else
  colormap(gca,b2r(-0.5,0.5)); colorbar; %caxis auto;
  title({strTitle,['JJA blocking frequency trends'],'%/yr'},'fontsize',16);
end
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERdjf_trend'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-25,25)); colorbar; %caxis auto;  %TODO
  title({['DJF weighted blocking frequency trends'],'unit*%/yr'},'fontsize',16);
else
  colormap(gca,b2r(-0.3,0.3)); colorbar; %caxis auto;
  title({['DJF blocking frequency trends'],'%/yr'},'fontsize',16);
end
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);

end  % textWgt

%% SVD spatial: xtrm_scatter
%{
textWgtArr = {'chunk','chunkwgt'};
for textWgt = textWgtArr

textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERdjf_xyn = squeeze(mean( reshape(PER_djf.*(Wgt_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
  Hot_xyn = squeeze(mean( reshape(Hot_jja.*(mx2t_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  Cold_xyn = squeeze(mean( reshape(Cold_djf.*(mn2t_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERdjf_xyn = squeeze(mean( reshape(PER_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
  Hot_xyn = squeeze(mean( reshape(Hot_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  Cold_xyn = squeeze(mean( reshape(Cold_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
end
PERjja_sn  =  reshape( PERjja_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
PERdjf_sn  =  reshape( PERdjf_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);
Hot_sn  =  reshape( Hot_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
Cold_sn  =  reshape( Cold_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);

% -mean
PERjja_sn = PERjja_sn -repmat(mean(PERjja_sn,2),[1 nyr]);  % double not needed..
PERdjf_sn = PERdjf_sn -repmat(mean(PERdjf_sn,2),[1 nyr-1]);  % double not needed..
Hot_sn = Hot_sn -repmat(mean(Hot_sn,2),[1 nyr]);  % double not needed..
Cold_sn = Cold_sn -repmat(mean(Cold_sn,2),[1 nyr-1]);  % double not needed..

%PER_yht(:,:,timeNan) = 0;

HotCov = Hot_sn * PERjja_sn';
ColdCov = Cold_sn * PERdjf_sn';

[HotU, HotS, HotV] = svds(double(HotCov),4);  % svd-subset. U:extreme; V:PER
[ColdU, ColdS, ColdV] = svds(double(ColdCov),4);  % svd-subset. U:extreme; V:PER

HotV = HotV * diag(sign(mean(HotU,1)));  HotU = HotU * diag(sign(mean(HotU,1)));
ColdV = ColdV * diag(sign(mean(ColdU,1)));  ColdU = ColdU * diag(sign(mean(ColdU,1)));
HotPrct1 = diag(HotS).^2/norm(HotCov,'fro')^2 *100;
HotPrct2 = sum(HotU,1).*diag(HotS)'.*sum(HotV,1) /sum(HotCov(:))*100;
ColdPrct1 = diag(ColdS).^2/norm(ColdCov,'fro')^2 *100;
ColdPrct2 = sum(ColdU,1).*diag(ColdS)'.*sum(ColdV,1) /sum(ColdCov(:))*100;
clear HotCov ColdCov;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for m=1:4
subplot(3,4,m,'ActivePositionProperty','position');
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, reshape(HotU(:,m),ds(1:2))'); shading flat;
colormap(gca,b2r(-0.15,0.15)); %colorbar;
plotm(coastlat,coastlon,'k')
%title({['Hot singular vector ',num2str(m)], ['square of singular value: ',num2str(HotPrct1(m),'%4.1f'),'%'], ['contrib. to scatter "r": ',num2str(HotPrct2(m),'%4.1f'),'%']},'fontsize',16);
title({['Hot #',num2str(m),', ',num2str(HotPrct1(m),'%4.1f'),'%, ',num2str(HotPrct2(m),'%4.1f'),'%']},'fontsize',16);
tightmap;

subplot(3,4,4+m,'ActivePositionProperty','position');
subplot(3,4,4+m,'ActivePositionProperty','position');  % bug..
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, reshape(HotV(:,m),ds(1:2))'); shading flat;
colormap(gca,b2r(-0.15,0.15)); %colorbar;
plotm(coastlat,coastlon,'k')
%title({['JJA blocking singular vector ',num2str(m)]},'fontsize',16);
title({['JJA blocking #',num2str(m)]},'fontsize',16);
tightmap;

subplot(3,4,8+m,'ActivePositionProperty','position');
yyaxis left;
plot([yStart:yEnd], sum(HotU(:,m),1) *HotU(:,m)' *Hot_sn, '-o');
xlim([yStart yEnd]); xlabel({'year'});
ax=gca; ylim(max(abs(ax.YLim))*[-1 1]);
ax.TickLength = [0.05 0.05];
yyaxis right;
plot([yStart:yEnd], sum(HotV(:,m),1) *HotV(:,m)' *PERjja_sn, '-o');
ax=gca; ylim(max(abs(ax.YLim))*[-1 1]);
%if (contains(textTH, 'wgt'))
%  ylabel({'Mean weighted blocking area (unit*km^2)'});
%else
%  ylabel({'Mean blocking area (km^2)'});
%end

end %m loop
if (contains(textTH, 'wgt'))
  subplot(3,4,9); yyaxis left; ylabel({'Mean weighted extreme area (K*km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean weighted blocking area (unit*km^2)'});
else
  subplot(3,4,9); yyaxis left; ylabel({'Mean extreme area (km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean blocking area (km^2)'});
end
print(gcf, '-dpsc2','-append',fn_figure);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for m=1:4
subplot(3,4,m,'ActivePositionProperty','position');
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, reshape(ColdU(:,m),ds(1:2))'); shading flat;
colormap(gca,b2r(-0.15,0.15)); %colorbar;
plotm(coastlat,coastlon,'k')
%title({['Cold singular vector ',num2str(m)], ['square of singular value: ',num2str(ColdPrct1(m),'%4.1f'),'%'], ['contrib. to scatter "r": ',num2str(ColdPrct2(m),'%4.1f'),'%']},'fontsize',16);
title({['Cold #',num2str(m),', ',num2str(ColdPrct1(m),'%4.1f'),'%, ',num2str(ColdPrct2(m),'%4.1f'),'%']},'fontsize',16);
tightmap;

subplot(3,4,4+m,'ActivePositionProperty','position');
subplot(3,4,4+m,'ActivePositionProperty','position');  % bug..
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, reshape(ColdV(:,m),ds(1:2))'); shading flat;
colormap(gca,b2r(-0.15,0.15)); %colorbar;
plotm(coastlat,coastlon,'k')
%title({['DJF blocking singular vector ',num2str(m)]},'fontsize',16);
title({['DJF blocking #',num2str(m)]},'fontsize',16);
tightmap;

subplot(3,4,8+m,'ActivePositionProperty','position');
yyaxis left;
plot([yStart+1:yEnd], sum(ColdU(:,m),1) *ColdU(:,m)' *Cold_sn, '-o');
xlim([yStart+1 yEnd]); xlabel({'year'});
ax=gca; ylim(max(abs(ax.YLim))*[-1 1]);
ax.TickLength = [0.05 0.05];
yyaxis right;
plot([yStart+1:yEnd], sum(ColdV(:,m),1) *ColdV(:,m)' *PERdjf_sn, '-o');
ax=gca; ylim(max(abs(ax.YLim))*[-1 1]);
%if (contains(textTH, 'wgt'))
%  ylabel({'Mean weighted blocking area (unit*km^2)'});
%else
%  ylabel({'Mean blocking area (km^2)'});
%end

end %m loop
if (contains(textTH, 'wgt'))
  subplot(3,4,9); yyaxis left; ylabel({'Mean weighted extreme area (K*km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean weighted blocking area (unit*km^2)'});
else
  subplot(3,4,9); yyaxis left; ylabel({'Mean extreme area (km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean blocking area (km^2)'});
end
print(gcf, '-dpsc2','-append',fn_figure);

end  % textWgt
%}


%}
system(['ps2pdf ',fn_figure]);
system(['rm ',fn_figure]);
toc
end  % exist rrr
%

%:'<,'>s/jja\C/djf/gce | '<,'>s/JJA\C/DJF/gce | '<,'>s/Hot\C/Cold/gce | '<,'>s/hot\C/cold/gce | '<,'>s/mx2t/mn2t/gce | '<,'>s/yStart:yEnd/yStart+1:yEnd/gce | '<,'>s/yyyy-yStart+1/yyyy-yStart/gce | '<,'>s?/nyr?/(nyr)?gce | '<,'>s/nyr/&-1/gce | noh

%:'<,'>s/nccreate(fn_savenc,'\([^']*\)',.*$/ncwrite(fn_savenc,'\1',\1)/ | noh

% : set fdm=expr foldexpr=getline(v\:lnum)=~'^%%.*$'?0\:1:
% vim: set fdm=marker foldmarker=%{,%}:

