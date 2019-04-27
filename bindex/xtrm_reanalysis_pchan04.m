%% define extremes, for F32 grid
% sbatch --account=kuang_lab -p huce_intel -J xtrm04 -n 1 -t 480 --mem=20000 -o "slurm" --wrap='\matlab -nodesktop -nodisplay -nosplash -r "xtrm_reanalysis_pchan04;exit" >& slurm-${SLURM_JOB_NAME}-${SLURM_JOBID}' --mail-type=END

tic;
warning('off','MATLAB:polyfit:RepeatedPointsOrRescale');
%% load and save data
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
% 1: number for quantile/sigma
% 2: 'quantile' or 'sigma'
% 3: persistence (d)
% 4: cutoff by temperature (K)
% 5: cutoff by latitude

fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
fn_load0 = ['../sks/int_z500_zg_day_',caseid,'N.nc'];  % 161021, 00Z
nc_load1 = @(fc) ['../ERA-interim/mx2t_12h_fc',num2str(fc),'_',caseid,'.nc'];
nc_load1b = @(fc) ['../ERA-interim/mn2t_12h_fc',num2str(fc),'_',caseid,'.nc'];
%fn_load2 = ['label_int_z500_',ver,'.mat'];
%fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
%fn_load2 = ['../sks/label_5000_ERA-interim_',ver,'.mat'];
%fn_pedram= ['../index_wise/Blocklabel_',ver,'.mat'];
%nc_load3 = ['../sks/vapvstar_6h_ERA-interim_19890101-20091231N.nc'];
nc_load9 = ['../ERA-interim/lsm_ERA-interim.nc'];
% fn_save  = ['composite_',ver,'.mat'];
%fn_savenc = ['temp_',textTH,'_',text,'.nc'];

%load(fn_load0,'Z500Daily')
yStart = str2num(caseid(end-16:end-13)); yEnd = str2num(caseid(end-7:end-4)); nyr = yEnd-yStart+1;
nDays = days(datetime(yEnd,12,31)-datetime(yStart,1,1)) +1;

f_h2d = @(h) hours(h)+datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S');
%time1 = ncread(nc_load1(find(fc==12)),'time',2,nDays-1,2);
time0 = ncread(fn_load0,'time');
mx2t = nan(128,64,nDays,'single');
mn2t = nan(128,64,nDays,'single');
ds = size(mx2t);
for fc = 9:3:18;
  wrk = ncread(nc_load1(fc),'mx2t',[1 1 2-(fc>12)],[Inf Inf 2*(ds(3)-1)]);
%  wrk = ncread(nc_load1(fc),'mx2t',[1 1 3-(fc>12)],[Inf Inf 2*(ds(3)-1)]);  %12Z
  mx2t(:,:,2:end) = max(mx2t(:,:,2:end), squeeze(max( reshape(wrk,[ds(1:2) 2 ds(3)-1]), [],3)) );
  wrk = ncread(nc_load1b(fc),'mn2t',[1 1 2-(fc>12)],[Inf Inf 2*(ds(3)-1)]);
%  wrk = ncread(nc_load1b(fc),'mn2t',[1 1 3-(fc>12)],[Inf Inf 2*(ds(3)-1)]);  %12Z
  mn2t(:,:,2:end) = min(mn2t(:,:,2:end), squeeze(min( reshape(wrk,[ds(1:2) 2 ds(3)-1]), [],3)) );
end
clear wrk

lsm = (ncread(nc_load9,'lsm')==1);  % double->logical, 2d

ver=['0602_',caseid];
fn_load2 = ['label_int_z500_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
load(fn_load2,'timeNan')
PER0202 = (mat_load2.label>0);

ver=['0702_',caseid];
fn_load2 = ['label_int_z500_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
%load(fn_load2,'timeNan')
PER0702 = (mat_load2.label>0);

ver=['2631_',caseid]; u=2;v=1;  %Hybrid2020, zprime02, bugfix+4
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
load(fn_load2,'Duration')
timeNan = unique([timeNan, 1:(Duration(v)-1), (ds(3)-Duration(v)+2):ds(3)]);
PER2225 = (squeeze(mat_load2.PER(:,:,:,u,v)));

ver=['2731_',caseid]; u=2;v=1;  %Hybrid2020, zprime07, bugfix+10
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
load(fn_load2,'Duration')
timeNan = unique([timeNan, 1:(Duration(v)-1), (ds(3)-Duration(v)+2):ds(3)]);
PER2731 = (squeeze(mat_load2.PER(:,:,:,u,v)));

ver=['4001_',caseid];  %NewReversal, bugfix+1
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
 %load(fn_load2,'timeNan')
PER4001 = (mat_load2.PER>=5);

fn_z500a  = ['../index_wise/Z500_06xx_',caseid,'.mat'];
mat_z500a = matfile(fn_z500a);
Z500a = mat_z500a.ZaDaily;  % already in latt42/lat0

%% check lat lon, referencing file0 (z500)
lat0  = ncread(fn_load0,'lat');
lat1  = ncread(nc_load1(9),'latitude');
lat1b  = ncread(nc_load1b(9),'latitude');
%lat2  = ncread(fn_load2,'latitude');
%lat3  = ncread(nc_load3,'latitude');
lat9  = ncread(nc_load9,'latitude');

lon0  = ncread(fn_load0,'lon');
lon1  = ncread(nc_load1(9),'longitude');
lon1b  = ncread(nc_load1b(9),'longitude');
%lon2  = ncread(fn_load2,'longitude');
%lon3  = ncread(nc_load3,'longitude');
lon9  = ncread(nc_load9,'longitude');

if (max(abs(lat0-lat1))>0.1)
  mx2t = mx2t(:,end:-1:1 ,:);
  lat1 = lat1(end:-1:1);
end
if (max(abs(lat0-lat1b))>0.1)
  mn2t = mn2t(:,end:-1:1 ,:);
  lat1b = lat1b(end:-1:1);
end
if (max(abs(lat0-lat9))>0.1)
  lsm = lsm(:,end:-1:1);
  lat9 = lat9(end:-1:1);
end
if (any(lon0<0) || max(abs(lat0-lat1))>0.1 || max(abs(lon0-lon1))>0.1 || max(abs(lat0-lat1b))>0.1 || max(abs(lon0-lon1b))>0.1 || max(abs(lat0-lat9))>0.1 || max(abs(lon0-lon9))>0.1)
  disp('error')
end

%if (any(lon2>180))
%  PER5000 = PER5000([181:360,1:180], :,:);
%  PER5001 = PER5001([181:360,1:180], :,:);
%  lon2 = lon2([181:360,1:180]);
%end
%if (any(lon3>180))
%  pv_jja = pv_jja([181:360,1:180], :,:);
%  lon3 = lon3([181:360,1:180]);
%end
%if (any(mod(lon0-lon1,360)) )
%  disp('error')
%end

%% define extremes
  wrk  = nanmean( double(mx2t),3); % double precision needed?
  mx2t = mx2t - repmat( wrk, [1 1 ds(3)]);
  wrk  = nanmean( double(mn2t),3); % double precision needed?
  mn2t = -(mn2t - repmat( wrk, [1 1 ds(3)]));

%  T850_yxht = zeros(ds(1),floor(ds(2)/2),2, ds(3),'single');
%  T850_yxht(:,:,1,:) = mx2t(:,1:floor(ds(2)/2)           ,:);
%  T850_yxht(:,:,2,:) = mx2t(:,end:-1:end-floor(ds(2)/2)+1,:);
%  T850_yxht = permute(T850_yxht,[2 1 3 4]);

%  T850avg_yh = zeros(floor(ds(2)/2),2,'single');
%  T850avg_yh(:,1) = wrk(1:floor(ds(2)/2));
%  T850avg_yh(:,2) = wrk(end:-1:end-floor(ds(2)/2)+1);
%clear wrk
%ncwrite(fn_savenc,'T850Daily',T850Daily)

% can use movmean instead..
mx2t = filter(ones(1,thresh{3}), thresh{3}, mx2t, [],3);
mx2t(:,:, (1+(thresh{3}-1)/2):(end-(thresh{3}-1)/2)) = mx2t(:,:,thresh{3}:end);
mx2t(:,:, [1:(thresh{3}-1)/2,end-(thresh{3}-1)/2+1:end]) = nan;
mn2t = filter(ones(1,thresh{3}), thresh{3}, mn2t, [],3);
mn2t(:,:, (1+(thresh{3}-1)/2):(end-(thresh{3}-1)/2)) = mn2t(:,:,thresh{3}:end);
mn2t(:,:, [1:(thresh{3}-1)/2,end-(thresh{3}-1)/2+1:end]) = nan;
timeNan = unique([timeNan, 1:(thresh{3}-1)/2,ds(3)-(thresh{3}-1)/2+1:ds(3)]);

% old, define extreme before collect JJA
%{
if (thresh{2}=='quantile')
  HotQuantile = quantile(mx2t, 1-thresh{1}, 3);
  Hot = ( mx2t> repmat( HotQuantile, [1 1 ds(3)]) );
  ColdQuantile = quantile(mn2t, 1-thresh{1}, 3);
  Cold = ( mn2t> repmat( ColdQuantile, [1 1 ds(3)]) );
elseif (thresh{2}=='sigma')
  wrk = zscore(reshape(permute(mx2t,[2 1 3]), ds(2),[]), [],2); %xyt->yxt
  TZscore = ipermute(reshape(wrk,ds([2 1 3])), [2 1 3]);
  Hot1d = single( TZscore>1.5 );
  Cold1d = single( TZscore<-1.5 );
  clear TZscore wrk
else
  disp('Please enter quantile or sigma. Exitting..')
  exit
end

  Hot( abs(mx2t) <thresh{4} ) =false;
  Cold( abs(mn2t) <thresh{4} ) =false;

  Hot = Hot & repmat(lsm, [1 1 ds(3)]);
  Cold = Cold & repmat(lsm, [1 1 ds(3)]);

  lsm_jja=lsm; lsm_djf=lsm;
%}

%% collect JJA
hJJAstart = hours(datetime(yStart:yEnd,6,1,0,0,0)  - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
hJJAend   = hours(datetime(yStart:yEnd,8,31,0,0,0) - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
ds_jja=ds; ds_jja(3) = sum(hJJAend-hJJAstart)/24 +numel(hJJAstart);

mx2t_jja = zeros(ds_jja,'single');
Z500a_jja = zeros(ds_jja,'single');
%Hot_jja = false(ds_jja);
PER0202_jja = false(ds_jja);
PER2225_jja = false(ds_jja);
PER4001_jja = false(ds_jja);
%pv_jja  = zeros(ds_jja,'single');
%PER5000 = false(ds_jja);
%PER5001 = false(ds_jja);
time_jja = zeros([ds_jja(3) 1], class(time0));
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time0==hJJAstart(yyyy-yStart+1));
  tend   = find(time0==hJJAend(yyyy-yStart+1));
  mx2t_jja(:,:,tpointer+(0:tend-tstart)) = mx2t(:,:,tstart:tend);
  Z500a_jja(:,:,tpointer+(0:tend-tstart)) = Z500a(:,:,tstart:tend);
%  Hot_jja(:,:,tpointer+(0:tend-tstart)) = Hot(:,:,tstart:tend);  % old: all season

  PER0202_jja(:,:,tpointer+(0:tend-tstart)) = PER0202(:,:,tstart:tend);
  PER2225_jja(:,:,tpointer+(0:tend-tstart)) = PER2225(:,:,tstart:tend);
  PER4001_jja(:,:,tpointer+(0:tend-tstart)) = PER4001(:,:,tstart:tend);

%  wrk = ncread(nc_load3, 'pv', [1 1 find(time3==(hJJAstart(yyyy-yStart+1)-3))], [Inf Inf tend-tstart+1+1]);
%  pv_jja(:,:,tpointer+(0:tend-tstart)) = ( wrk(:,:,1:end-1) + wrk(:,:,2:end) )/2;

%  fn_load2 = ['../sks/label_5000_ERA-interim_',num2str(yyyy),'jjaN.nc'];
%  time_PER = ncread(fn_load2,'time');
%  wrk = ncread(fn_load2, 'label', [1 1 find(time_PER==(hJJAstart(yyyy-yStart+1)-3))], [Inf Inf tend-tstart+1+1]);
%  PER5000(:,:,tpointer+(0:tend-tstart)) = ( wrk(:,:,1:end-1)>0 | wrk(:,:,2:end)>0 );
%
%  fn_load2 = ['../sks/label_5001_ERA-interim_',num2str(yyyy),'jjaN.nc'];
%  time_PER = ncread(fn_load2,'time');
%  wrk = ncread(fn_load2, 'label', [1 1 find(time_PER==(hJJAstart(yyyy-yStart+1)-3))], [Inf Inf tend-tstart+1+1]);
%  PER5001(:,:,tpointer+(0:tend-tstart)) = ( wrk(:,:,1:end-1)>0 | wrk(:,:,2:end)>0 );

  time_jja(tpointer+(0:tend-tstart)) = time0(tstart:tend);
  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
hDJFstart = hours(datetime(yStart:yEnd-1,12,1,0,0,0)  - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
hDJFend   = hours(datetime(yStart+1:yEnd, 2,28,0,0,0) - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
ds_djf=ds; ds_djf(3) = sum(hDJFend-hDJFstart)/24 +numel(hDJFstart);

mn2t_djf = zeros(ds_djf,'single');
Z500a_djf = zeros(ds_djf,'single');
%Cold_djf = false(ds_djf);
PER0202_djf = false(ds_djf);
PER2225_djf = false(ds_djf);
PER4001_djf = false(ds_djf);
time_djf = zeros([ds_djf(3) 1], class(time0));
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time0==hDJFstart(yyyy-yStart));
  tend   = find(time0==hDJFend(yyyy-yStart));
  mn2t_djf(:,:,tpointer+(0:tend-tstart)) = mn2t(:,:,tstart:tend);
  Z500a_djf(:,:,tpointer+(0:tend-tstart)) = Z500a(:,:,tstart:tend);
%  Cold_djf(:,:,tpointer+(0:tend-tstart)) = Cold(:,:,tstart:tend);  % old: all season

  PER0202_djf(:,:,tpointer+(0:tend-tstart)) = PER0702(:,:,tstart:tend);
  PER2225_djf(:,:,tpointer+(0:tend-tstart)) = PER2731(:,:,tstart:tend);
  PER4001_djf(:,:,tpointer+(0:tend-tstart)) = PER4001(:,:,tstart:tend);

  time_djf(tpointer+(0:tend-tstart)) = time0(tstart:tend);
  tpointer = tpointer +tend-tstart+1;
end
clear mx2t mn2t Z500a Hot Cold PER0202 PER2225 PER4001 PER0702 PER2731  yyyy tpointer tstart tend lat9 lon9 mat_z500a mat_load2 fn_load2 u v fc

%% remove trend
mx2t_xyn = squeeze(mean( reshape(mx2t_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
mn2t_xyn = squeeze(mean( reshape(mn2t_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr

%
lsm_jja = lsm &(mean(mx2t_xyn,3)>0);  % found some place have JJA mean temperature smaller than all season mean
%lsm_jja = lsm;
%{
fn_figure = ['meanJJA_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);

jArr = find(lat0>0); jArr = jArr(1:2:end); %set(groot,'defaultAxesColorOrder',hsv(length(jArr)));
figure('paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]); hold on; ax = gca;
ax.ColorOrder = hsv(length(jArr));
%ax.ColorOrder = lines(length(jArr));
%for j=jArr(:)'
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_jja(:,j));
  if (nLand>0)
    iArr = find(lsm_jja(:,j));
    i = iArr(randi(nLand));
    ax.ColorOrderIndex = jj;
%    plot(yStart:yEnd, squeeze(mx2t_xyn(i,j,:)));
%    ax.ColorOrderIndex = jj;
%    plot(yStart:yEnd, movmean( squeeze(mx2t_xyn(i,j,:)), 5), '--');
    plot(yStart:yEnd, jj*3-mean(mx2t_xyn(i,j,1:3)) +squeeze(mx2t_xyn(i,j,:)));
    ax.ColorOrderIndex = jj;
    plot(yStart:yEnd, jj*3-mean(mx2t_xyn(i,j,1:3)) +movmean( squeeze(mx2t_xyn(i,j,:)), 5), '--');
  end
end
print(gcf, '-dpsc2','-append',fn_figure);

system(['ps2pdf ',fn_figure]);
%}

lsm_djf = lsm &(mean(mn2t_xyn,3)>0);  % basically northern hemisphere
%lsm_djf = lsm;
%{
fn_figure = ['meanDJF_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);

jArr = find(lat0>0); jArr = jArr(1:2:end); %set(groot,'defaultAxesColorOrder',hsv(length(jArr)));
figure('paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]); hold on; ax = gca;
%ax.ColorOrder = hsv(length(jArr));
ax.ColorOrder = lines(length(jArr));
%for j=jArr(:)'
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_djf(:,j));
  if (nLand>0)
    iArr = find(lsm_djf(:,j));
    i = iArr(randi(nLand));
    ax.ColorOrderIndex = jj;
%    plot(yStart+1:yEnd, -squeeze(mn2t_xyn(i,j,:)));
%    ax.ColorOrderIndex = jj;
%    plot(yStart+1:yEnd, -movmean( squeeze(mn2t_xyn(i,j,:)), 5), '--');
    plot(yStart+1:yEnd, jj*3+mean(mn2t_xyn(i,j,1:3)) -squeeze(mn2t_xyn(i,j,:)));
    ax.ColorOrderIndex = jj;
    plot(yStart+1:yEnd, jj*3+mean(mn2t_xyn(i,j,1:3)) -movmean( squeeze(mn2t_xyn(i,j,:)), 5), '--');
  end
end
print(gcf, '-dpsc2','-append',fn_figure);

system(['ps2pdf ',fn_figure]);
%}

mx2t_jja = mx2t_jja - reshape(repmat(reshape(movmean(mx2t_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
mn2t_djf = mn2t_djf - reshape(repmat(reshape(movmean(mn2t_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);

if (thresh{2}=='quantile')
  HotQuantile = quantile(mx2t_jja, 1-thresh{1}, 3);
  Hot_jja = ( mx2t_jja> repmat( HotQuantile, [1 1 ds_jja(3)]) );
  ColdQuantile = quantile(mn2t_djf, 1-thresh{1}, 3);
  Cold_djf = ( mn2t_djf> repmat( ColdQuantile, [1 1 ds_djf(3)]) );
elseif (thresh{2}=='sigma')
  clear TZscore wrk
else
  disp('Please enter quantile or sigma. Exitting..')
  exit
end

  Hot_jja( abs(mx2t_jja) <thresh{4} ) =false;
  Cold_djf( abs(mn2t_djf) <thresh{4} ) =false;

  Hot_jja = Hot_jja & repmat(lsm_jja, [1 1 ds_jja(3)]);
  Cold_djf = Cold_djf & repmat(lsm_djf, [1 1 ds_djf(3)]);
%

save(['temp_',textTH,'_',text,'.mat'],'-v7.3')
%}

%% plot quantile (cf xtrmfreq)
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh' ,'lat0','lon0' ,'lsm_jja','HotQuantile' ,'lsm_djf','ColdQuantile')

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
lonplot = [lon0(:); 2*lon0(end)-lon0(end-1)];  % cyclic point added

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,1,1);
%axesm('MapProjection','ortho','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20)
%axesm('MapProjection','vperspec','origin',[90 0],'MapParallels',6000,'grid','on','mlinelocation',20,'plinelocation',20);
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
HotQuantile(~lsm_jja) = nan;
%[cc,hc]=contourfm(lat1a,lon1a,double(HotQuantile'));
%pcolorm(double(lat0),double(lon0([end/2:end,1:end/2])),double(HotQuantile([end/2:end,1:end/2],:)')); shading flat;  % cyclic point added
pcolormPC(lat0,lon0,HotQuantile'); shading flat;
%contourm(lat0,lonplot,double(HotQuantile([1:end,1],:))',[1:12]); axis equal tight;
%clabelm(cc,hc,[-2:1:1,1.2])%,'BackgroundColor','none')
%colormap(jet(10)); caxis([0 40]); colorbar;  % old: all season
colormap(jet(12)); caxis([0 12]); colorbar;
plotm(coastlat,coastlon,'k')
%title('\fontsize{20}The 99th percentile of six-hourly maximum temperature T_{max}');
%title({'99th percentile of daily T_{max}','(all season mean removed)'},'fontsize',16);  % old: all season
title({strTitle,'99th percentile of daily T_{max}','(JJA mean removed)'},'fontsize',16);
xlim([-pi pi]); ylim([0 pi/2]);  % tightmap??

subplot(2,1,2);
%axesm('MapProjection','ortho','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20)
%axesm('MapProjection','vperspec','origin',[90 0],'MapParallels',6000,'grid','on','mlinelocation',20,'plinelocation',20);
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
ColdQuantile(~lsm_djf) = nan;
%[cc,hc]=contourfm(lat1a,lon1a,double(-ColdQuantile'));
%pcolorm(double(lat0),double(lon0([end/2:end,1:end/2])),double(-ColdQuantile([end/2:end,1:end/2],:)')); shading flat;  % cyclic point added
pcolormPC(lat0,lon0,-ColdQuantile'); shading flat;
%contourm(lat0,lonplot,double(-ColdQuantile([1:end,1],:))',[-24:2:-2]); axis equal tight;
%clabelm(cc,hc,[-2:1:1,1.2])%,'BackgroundColor','none')
%colormap(jet(10)); caxis([-40 0]); colorbar;  % old: all season
colormap(jet(12)); caxis([-24 0]); colorbar;
plotm(coastlat,coastlon,'k')
%title({'1st percentile of daily T_{min}','(all season mean removed)'},'fontsize',16);  % old: all season
title({'1st percentile of daily T_{min}','(DJF mean removed)'},'fontsize',16);
xlim([-pi pi]); ylim([0 pi/2]);  % tightmap??

savefig(gcf,['quantile_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['quantile_',textTH,'_',text,'.pdf'])  %xtrmfreq
%}

% Seasonal Cycle
%{
i=30; j=42;
figure('outerPosition',[0 40 1920 1040],'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
plot(reshape( squeeze(mx2t_jja(i,j,:)),[],nyr ));
plot([1 92],HotQuantile(i,j)*[1 1],'k-');
title([num2str(lat0(j),2),'N ',num2str(lon0(i),3),'E']);
savefig(gcf,['SeasonalCycleIndiaJJA_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['SeasonalCycleIndiaJJA_',textTH,'_',text,'.pdf'])

%fn_figure = 'SeasonalCycle.ps';
%system(['rm ',fn_figure]);
figure('outerPosition',[0 40 1920 1040],'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]); hold on;
jArr = find(lat0>0); %jArr = jArr(1:2:end);
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_jja(:,j));
  if (nLand>0)
    iArr = find(lsm_jja(:,j));
    i = iArr(randi(nLand));
    subplot(5,6,jj); hold on;
    ax=gca; ax.ColorOrder = jet(nyr);
    plot(reshape( squeeze(mx2t_jja(i,j,:)),[],nyr ));
    plot([1 92],HotQuantile(i,j)*[1 1],'k-');
    title([num2str(lat0(j),2),'N ',num2str(lon0(i),3),'E']);
%    print(gcf, '-dpsc2','-append',fn_figure);
  end
end
%system(['ps2pdf ',fn_figure]);
savefig(gcf,['SeasonalCycleJJA_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf','-r600',['SeasonalCycleJJA_',textTH,'_',text,'.pdf'])

figure('outerPosition',[0 40 1920 1040],'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]); hold on;
jArr = find(lat0>0); %jArr = jArr(1:2:end);
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_djf(:,j));
  if (nLand>0)
    iArr = find(lsm_djf(:,j));
    i = iArr(randi(nLand));
    subplot(5,6,jj); hold on;
    ax=gca; ax.ColorOrder = jet(nyr-1);
    plot(reshape( squeeze(-mn2t_djf(i,j,:)),[],nyr-1 ));
    plot([1 90],-ColdQuantile(i,j)*[1 1],'k-');
    title([num2str(lat0(j),2),'N ',num2str(lon0(i),3),'E']);
  end
end
savefig(gcf,['SeasonalCycleDJF_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])
%}

%% other index are loaded here..
%{
%  PER0202_yxht = false(ds(1),floor(ds(2)/2),2, ds(3));
%  PER0202_yxht(:,:,1,:) = PER0202(:,1:floor(ds(2)/2)           ,:);
%  PER0202_yxht(:,:,2,:) = PER0202(:,end:-1:end-floor(ds(2)/2)+1,:);
%  PER0202_yxht = permute(PER0202_yxht,[2 1 3 4]);
%  clear PER0202
%
%  PER2022_yxht = false(ds(1),floor(ds(2)/2),2, ds(3));
%  PER2022_yxht(:,:,1,:) = PER2022(:,1:floor(ds(2)/2)           ,:);
%  PER2022_yxht(:,:,2,:) = PER2022(:,end:-1:end-floor(ds(2)/2)+1,:);
%  PER2022_yxht = permute(PER2022_yxht,[2 1 3 4]);
%  clear PER2022
%
%  PER4001_yxht = false(ds(1),floor(ds(2)/2),2, ds(3));
%  PER4001_yxht(:,:,1,:) = PER4001(:,1:floor(ds(2)/2)           ,:);
%  PER4001_yxht(:,:,2,:) = PER4001(:,end:-1:end-floor(ds(2)/2)+1,:);
%  PER4001_yxht = permute(PER4001_yxht,[2 1 3 4]);
%  clear PER4001

%% prepare nc
%  lat = ncread(fn_t42,'lat');
%  lat2 = abs( lat(1:floor(ds(2)/2)) );  % only hemisphere

% system(['rm ',fn_savenc]);
%system(['ncks -6 -v lat,lat_bnds,lon,lon_bnds ',fn_t42,' ',fn_savenc]);
%nccreate(fn_savenc,'T850Daily','Dimensions',{'lon',128,'lat',64,'time',ds(3)},'DataType','single')
%nccreate(fn_savenc,'Hot','Dimensions',{'lon',128,'lat',64,'time',ds(3)},'DataType','int8')
%nccreate(fn_savenc,'Cold','Dimensions',{'lon',128,'lat',64,'time',ds(3)},'DataType','int8')

%% other comments..
%ncwrite(fn_savenc,'Hot',int8(Hot))
%ncwrite(fn_savenc,'Cold',int8(Cold))

% Label only the center of the days. only forward cycle needed
%  hotDays(:,:,1)  = single( Hot1d(:,:,1) );
%  coldDays(:,:,1) = single( Cold1d(:,:,1) );
%  for t = 2:ds(3)
%      hotDays(:,:,t)  = (hotDays(:,:,t-1)+1) .* (Hot1d (:,:,t));
%      coldDays(:,:,t) = (coldDays(:,:,t-1)+1).* (Cold1d(:,:,t));
%  end
%
%  Hot5d  = nan(ds,'single');
%  Cold5d = nan(ds,'single');
%  Hot5d (:,:,3:end-2) = hotDays (:,:,5:end)>=5;
%  Cold5d(:,:,3:end-2) = coldDays(:,:,5:end)>=5;

%  Hot_yxht = false(ds(1),floor(ds(2)/2),2, ds(3));
%  Hot_yxht(:,:,1,:) = Hot(:,1:floor(ds(2)/2)           ,:);
%  Hot_yxht(:,:,2,:) = Hot(:,end:-1:end-floor(ds(2)/2)+1,:);
%  Hot_yxht = permute(Hot_yxht,[2 1 3 4]);
%
%  Cold_yxht = false(ds(1),floor(ds(2)/2),2, ds(3));
%  Cold_yxht(:,:,1,:) = Cold(:,1:floor(ds(2)/2)           ,:);
%  Cold_yxht(:,:,2,:) = Cold(:,end:-1:end-floor(ds(2)/2)+1,:);
%  Cold_yxht = permute(Cold_yxht,[2 1 3 4]);
%
%  Z500_yxht = zeros(ds(1),floor(ds(2)/2),2, ds(3),'single');
%  Z500_yxht(:,:,1,:) = Z500Daily(:,1:floor(ds(2)/2)           ,:);
%  Z500_yxht(:,:,2,:) = Z500Daily(:,end:-1:end-floor(ds(2)/2)+1,:);
%  Z500_yxht = permute(Z500_yxht,[2 1 3 4]);
%
%  clear hotDays coldDays Hot Cold Z500Daily T850Daily

%save(['temp_',textTH,'_',text,'.mat'])
%system(['ln -sf temp_',textTH,'_',text,'.mat temp_',text,'.mat']);

%fn_savenc = ['temp_',text,'.nc'];
%
% system(['rm ',fn_savenc]);
 %system(['ncks -6 -v lat,lat_bnds,lon,lon_bnds ',fn_t42,' ',fn_savenc]);
%system(['ncks -6 -v lon,lon_bnds ',fn_t42,' ',fn_savenc]);
%nccreate(fn_savenc,'lat2','Dimensions',{'lat2',floor(ds(2)/2)},'DataType','single')
%nccreate(fn_savenc,'T850_yxht','Dimensions',{'lat2',floor(ds(2)/2),'lon',ds(1),'hemi',2,'time',ds(3)},'DataType','single')
%nccreate(fn_savenc,'Hot_yxht','Dimensions',{'lat2',floor(ds(2)/2),'lon',ds(1),'hemi',2,'time',ds(3)},'DataType','single')
%nccreate(fn_savenc,'Cold_yxht','Dimensions',{'lat2',floor(ds(2)/2),'lon',ds(1),'hemi',2,'time',ds(3)},'DataType','single')
%ncwriteatt(fn_savenc,'lat2','units','degrees_north')
%ncwrite(fn_savenc,'lat2',lat2)
%ncwrite(fn_savenc,'T850_yxht',T850_yxht)
%ncwrite(fn_savenc,'Hot_yxht',Hot_yxht)
%ncwrite(fn_savenc,'Cold_yxht',Cold_yxht)
%}

%% time shape, xtrm_scatter_pchan
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh' ,'ds','lat0','lon0','yStart','yEnd','nyr','time0' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend','lsm_jja' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend','lsm_djf');
load(['temp_',textTH,'_',text,'.mat'],'ver');

fn_tend = ['../ncl/daily6h_',caseid,'.nc'];

T850  = ncread(fn_tend,'T850Daily');
W850  = ncread(fn_tend,'W850Daily');
T850z = ncread(fn_tend,'S850Daily');
T850AdvU = ncread(fn_tend,'T850AdvU');
T850AdvV = ncread(fn_tend,'T850AdvV');
T850AdvW = ncread(fn_tend,'T850AdvW');
T850err  = ncread(fn_tend,'T850err');

%% check lat lon, referencing file0 (z500)
% time0
latncl = ncread(fn_tend,'lat');
lonncl = ncread(fn_tend,'lon');

if (max(abs(lat0-latncl))>0.1)
  T850 = T850(:,end:-1:1 ,:);
  W850 = W850(:,end:-1:1 ,:);
  T850z = T850z(:,end:-1:1 ,:);
  T850AdvU = T850AdvU(:,end:-1:1 ,:);
  T850AdvV = T850AdvV(:,end:-1:1 ,:);
  T850AdvW = T850AdvW(:,end:-1:1 ,:);
  T850err = T850err(:,end:-1:1 ,:);

  latncl = latncl(end:-1:1);
end
if (max(abs(lat0-latncl))>0.1 || max(abs(lon0-lonncl))>0.1)
  disp('error')
end

%% collect JJA
T850_jja = zeros(ds_jja,'single');
W850_jja = zeros(ds_jja,'single');
T850z_jja = zeros(ds_jja,'single');
T850AdvU_jja = zeros(ds_jja,'single');
T850AdvV_jja = zeros(ds_jja,'single');
T850AdvW_jja = zeros(ds_jja,'single');
T850err_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time0==hJJAstart(yyyy-yStart+1));
  tend   = find(time0==hJJAend(yyyy-yStart+1));
  T850_jja(:,:,tpointer+(0:tend-tstart)) = T850(:,:,tstart:tend);
  W850_jja(:,:,tpointer+(0:tend-tstart)) = W850(:,:,tstart:tend);
  T850z_jja(:,:,tpointer+(0:tend-tstart)) = T850z(:,:,tstart:tend);
  T850AdvU_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvU(:,:,tstart:tend);
  T850AdvV_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvV(:,:,tstart:tend);
  T850AdvW_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvW(:,:,tstart:tend);
  T850err_jja(:,:,tpointer+(0:tend-tstart)) = T850err(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
T850_djf = zeros(ds_djf,'single');
W850_djf = zeros(ds_djf,'single');
T850z_djf = zeros(ds_djf,'single');
T850AdvU_djf = zeros(ds_djf,'single');
T850AdvV_djf = zeros(ds_djf,'single');
T850AdvW_djf = zeros(ds_djf,'single');
T850err_djf = zeros(ds_djf,'single');
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time0==hDJFstart(yyyy-yStart));
  tend   = find(time0==hDJFend(yyyy-yStart));
  T850_djf(:,:,tpointer+(0:tend-tstart)) = T850(:,:,tstart:tend);
  W850_djf(:,:,tpointer+(0:tend-tstart)) = W850(:,:,tstart:tend);
  T850z_djf(:,:,tpointer+(0:tend-tstart)) = T850z(:,:,tstart:tend);
  T850AdvU_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvU(:,:,tstart:tend);
  T850AdvV_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvV(:,:,tstart:tend);
  T850AdvW_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvW(:,:,tstart:tend);
  T850err_djf(:,:,tpointer+(0:tend-tstart)) = T850err(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end

%% remove trend
T850jja_xyn = squeeze(mean( reshape(T850_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
W850jja_xyn = squeeze(mean( reshape(W850_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%T850zjja_xyn = squeeze(mean( reshape(T850z_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850AdvUjja_xyn = squeeze(mean( reshape(T850AdvU_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850AdvVjja_xyn = squeeze(mean( reshape(T850AdvV_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850AdvWjja_xyn = squeeze(mean( reshape(T850AdvW_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850errjja_xyn = squeeze(mean( reshape(T850err_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr

T850djf_xyn = squeeze(mean( reshape(T850_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
W850djf_xyn = squeeze(mean( reshape(W850_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%T850zdjf_xyn = squeeze(mean( reshape(T850z_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850AdvUdjf_xyn = squeeze(mean( reshape(T850AdvU_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850AdvVdjf_xyn = squeeze(mean( reshape(T850AdvV_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850AdvWdjf_xyn = squeeze(mean( reshape(T850AdvW_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850errdjf_xyn = squeeze(mean( reshape(T850err_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr

T850_jja = T850_jja - reshape(repmat(reshape(movmean(T850jja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
W850_jja = W850_jja - reshape(repmat(reshape(movmean(W850jja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
%T850z_jja = T850z_jja - reshape(repmat(reshape(movmean(T850zjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850AdvU_jja = T850AdvU_jja - reshape(repmat(reshape(movmean(T850AdvUjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850AdvV_jja = T850AdvV_jja - reshape(repmat(reshape(movmean(T850AdvVjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850AdvW_jja = T850AdvW_jja - reshape(repmat(reshape(movmean(T850AdvWjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850err_jja = T850err_jja - reshape(repmat(reshape(movmean(T850errjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);

T850_djf = T850_djf - reshape(repmat(reshape(movmean(T850djf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
W850_djf = W850_djf - reshape(repmat(reshape(movmean(W850djf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
%T850z_djf = T850z_djf - reshape(repmat(reshape(movmean(T850zdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850AdvU_djf = T850AdvU_djf - reshape(repmat(reshape(movmean(T850AdvUdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850AdvV_djf = T850AdvV_djf - reshape(repmat(reshape(movmean(T850AdvVdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850AdvW_djf = T850AdvW_djf - reshape(repmat(reshape(movmean(T850AdvWdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850err_djf = T850err_djf - reshape(repmat(reshape(movmean(T850errdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);

clear T850 W850 T850z T850AdvU T850AdvV T850AdvW T850err

%% land
%wrk = repmat(lsm_jja, [1 1 ds_jja(3)]);
%T850AdvU_jja(~wrk) = nan;

T850tend_jja = T850AdvU_jja+T850AdvV_jja+T850AdvW_jja+T850err_jja;
T850tend_djf = T850AdvU_djf+T850AdvV_djf+T850AdvW_djf+T850err_djf;


%HotT_jja = nan(floor(ds(2)/2),ds(1),2, ds(3),'single');
%    HotT_jja(:,:,:,7)  = (T850_jja(:,:,:,7)) .*(Hot_jja(:,:,:,7));
%    for t = 8:ds(3)-6
%        HotT_jja(:,:,:,t) = max(HotT_jja(:,:,:,t-1),T850_jja(:,:,:,t)).* (Hot_jja(:,:,:,t));
%    end
%    for t = ds(3)-7 : -1 :7
%        HotT_jja(:,:,:,t) = max(HotT_jja(:,:,:,t),HotT_jja(:,:,:,t+1)).* (Hot_jja(:,:,:,t));
%    end

Hot_cp = reshape(Hot_jja, ds(1),ds(2),[],nyr );
Hot_cp(:,:,[1:6, end-5:end],:) = false;  % used in place of hotT ==T850
Hot_cp(:,lat0(:)<=30,:,:) = false;
ind = find(Hot_cp(:));

HotTs_t = zeros(13,1,'single');
HotT_t = zeros(13,1,'single');
HotW_t = zeros(13,1,'single');
HotDT_t = zeros(13,1,'single');
HotAdvU_t = zeros(13,1,'single');
HotAdvV_t = zeros(13,1,'single');
HotAdvW_t = zeros(13,1,'single');
Hoterr_t = zeros(13,1,'single');
for it = -6:6;
    HotTs_t(it+7) = mean(mx2t_jja(ind + ds(1)*ds(2)*it));
    HotT_t(it+7) = mean(T850_jja(ind + ds(1)*ds(2)*it));
    HotW_t(it+7) = mean(W850_jja(ind + ds(1)*ds(2)*it));
    HotDT_t(it+7) = mean(T850z_jja(ind + ds(1)*ds(2)*it));
    HotAdvU_t(it+7) = mean(T850AdvU_jja(ind + ds(1)*ds(2)*it));
    HotAdvV_t(it+7) = mean(T850AdvV_jja(ind + ds(1)*ds(2)*it));
    HotAdvW_t(it+7) = mean(T850AdvW_jja(ind + ds(1)*ds(2)*it));
    Hoterr_t(it+7) = mean(T850err_jja(ind + ds(1)*ds(2)*it));
end

%{
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,2,1)
plot( -6:6, HotT_t,'-o')
%hold on
%plot( -5.5:5.5, HotT_t(2:end)-HotT_t(1:end-1))
xlabel('time (day)');ylabel('T850 Anomaly (K)');grid on;
title('composite w.r.t. extreme hot event')

subplot(2,2,2)
plot( -6:6, HotW_t,'-o')
xlabel('time (day)');ylabel('\omega850 anomaly (Pa/s)');grid on;
%xlabel('time (day)');ylabel('poleward V850 Anomaly (m/s)');grid on;

subplot(2,2,3)
plot( -6:6, HotDT_t,'-o')
xlabel('time (day)');ylabel('Static stability (K/Pa)');grid on;
%xlabel('time (day)');ylabel('\partial T/\partial y (K/m)');grid on;

subplot(2,2,4)
plot( -6:6,HotAdvU_t*86400,'c-o', -6:6,HotAdvV_t*86400,'b-v', -6:6,HotAdvW_t*86400,'r-o', -6:6,(HotAdvU_t+HotAdvV_t+HotAdvW_t)*86400,'k-o', -6:6,(Hoterr_t)*86400,'g-o')
legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal 3d Adv','anomal Error');
xlabel('time (day)');ylabel('T850 tendency (K/day)');grid on;

savefig(gcf,['HotTimeEvolution_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['HotTimeEvolution_',textTH,'_',text,'.pdf'])
%}


%ColdT_djf = nan(floor(ds(2)/2),ds(1),2, ds(3),'single');
%    ColdT_djf(:,:,:,7)  = (T850_djf(:,:,:,7)) .*(Cold_djf(:,:,:,7));
%    for t = 8:ds(3)-6
%        ColdT_djf(:,:,:,t) = max(ColdT_djf(:,:,:,t-1),T850_djf(:,:,:,t)).* (Cold_djf(:,:,:,t));
%    end
%    for t = ds(3)-7 : -1 :7
%        ColdT_djf(:,:,:,t) = max(ColdT_djf(:,:,:,t),ColdT_djf(:,:,:,t+1)).* (Cold_djf(:,:,:,t));
%    end

Cold_cp = reshape(Cold_djf, ds(1),ds(2),[],nyr-1 );
Cold_cp(:,:,[1:6, end-5:end],:) = false;  % used in place of coldT ==T850
Cold_cp(:,lat0(:)<=30,:,:) = false;
ind = find(Cold_cp(:));

ColdTs_t = zeros(13,1,'single');
ColdT_t = zeros(13,1,'single');
ColdW_t = zeros(13,1,'single');
ColdDT_t = zeros(13,1,'single');
ColdAdvU_t = zeros(13,1,'single');
ColdAdvV_t = zeros(13,1,'single');
ColdAdvW_t = zeros(13,1,'single');
Colderr_t = zeros(13,1,'single');
for it = -6:6;
    ColdTs_t(it+7) = mean(mn2t_djf(ind + ds(1)*ds(2)*it));
    ColdT_t(it+7) = mean(T850_djf(ind + ds(1)*ds(2)*it));
    ColdW_t(it+7) = mean(W850_djf(ind + ds(1)*ds(2)*it));
    ColdDT_t(it+7) = mean(T850z_djf(ind + ds(1)*ds(2)*it));
    ColdAdvU_t(it+7) = mean(T850AdvU_djf(ind + ds(1)*ds(2)*it));
    ColdAdvV_t(it+7) = mean(T850AdvV_djf(ind + ds(1)*ds(2)*it));
    ColdAdvW_t(it+7) = mean(T850AdvW_djf(ind + ds(1)*ds(2)*it));
    Colderr_t(it+7) = mean(T850err_djf(ind + ds(1)*ds(2)*it));
end

%
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,2,1)
plot( -6:6, ColdT_t,'-o')
%hold on
%plot( -5.5:5.5, ColdT_t(2:end)-ColdT_t(1:end-1))
xlabel('time (day)');ylabel('T850 Anomaly (K)');grid on;
title('composite w.r.t. extreme cold event')

subplot(2,2,2)
plot( -6:6, ColdW_t,'-o')
xlabel('time (day)');ylabel('\omega850 anomaly (Pa/s)');grid on;
%xlabel('time (day)');ylabel('poleward V850 Anomaly (m/s)');grid on;

subplot(2,2,3)
plot( -6:6, ColdDT_t,'-o')
xlabel('time (day)');ylabel('Static stability (K/Pa)');grid on;
%xlabel('time (day)');ylabel('\partial T/\partial y (K/m)');grid on;

subplot(2,2,4)
plot( -6:6,ColdAdvU_t*86400,'c-o', -6:6,ColdAdvV_t*86400,'b-v', -6:6,ColdAdvW_t*86400,'r-o', -6:6,(ColdAdvU_t+ColdAdvV_t+ColdAdvW_t)*86400,'k-o', -6:6,(Colderr_t)*86400,'g-o')
legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal 3d Adv','anomal Error','location','southeast');
xlabel('time (day)');ylabel('T850 tendency (K/day)');grid on;

savefig(gcf,['ColdTimeEvolution_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['ColdTimeEvolution_',textTH,'_',text,'.pdf'])
%

%% tendency regress
% no need reshape, no nan accepted
krErr_y = nan([ds(2) 2]);  % so-called real component
kcErr_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krErr_y(y,:) = polyfit((T850_jja(:,y,:)),(T850err_jja(:,y,:)),1);  % no need double
%  kcErr_y(y,:) = polyfit((T850tend_jja(:,y,:)),(T850err_jja(:,y,:)),1);  % no need double
  wrk = lsm_jja(:,y);
  krErr_y(y,:) = polyfit((T850_jja(wrk,y,:)),(T850err_jja(wrk,y,:)),1);  % no need double
  kcErr_y(y,:) = polyfit((T850tend_jja(wrk,y,:)),(T850err_jja(wrk,y,:)),1);  % no need double
end
krErr_y = krErr_y(:,1)*86400;
kcErr_y = kcErr_y(:,1);

krAdvU_y = nan([ds(2) 2]);  % so-called real component
kcAdvU_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krAdvU_y(y,:) = polyfit((T850_jja(:,y,:)),(T850AdvU_jja(:,y,:)),1);  % no need double
%  kcAdvU_y(y,:) = polyfit((T850tend_jja(:,y,:)),(T850AdvU_jja(:,y,:)),1);  % no need double
  wrk = lsm_jja(:,y);
  krAdvU_y(y,:) = polyfit((T850_jja(wrk,y,:)),(T850AdvU_jja(wrk,y,:)),1);  % no need double
  kcAdvU_y(y,:) = polyfit((T850tend_jja(wrk,y,:)),(T850AdvU_jja(wrk,y,:)),1);  % no need double
end
krAdvU_y = krAdvU_y(:,1)*86400;
kcAdvU_y = kcAdvU_y(:,1);

krAdvV_y = nan([ds(2) 2]);  % so-called real component
kcAdvV_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krAdvV_y(y,:) = polyfit((T850_jja(:,y,:)),(T850AdvV_jja(:,y,:)),1);  % no need double
%  kcAdvV_y(y,:) = polyfit((T850tend_jja(:,y,:)),(T850AdvV_jja(:,y,:)),1);  % no need double
  wrk = lsm_jja(:,y);
  krAdvV_y(y,:) = polyfit((T850_jja(wrk,y,:)),(T850AdvV_jja(wrk,y,:)),1);  % no need double
  kcAdvV_y(y,:) = polyfit((T850tend_jja(wrk,y,:)),(T850AdvV_jja(wrk,y,:)),1);  % no need double
end
krAdvV_y = krAdvV_y(:,1)*86400;
kcAdvV_y = kcAdvV_y(:,1);

krAdvW_y = nan([ds(2) 2]);  % so-called real component
kcAdvW_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krAdvW_y(y,:) = polyfit((T850_jja(:,y,:)),(T850AdvW_jja(:,y,:)),1);  % no need double
%  kcAdvW_y(y,:) = polyfit((T850tend_jja(:,y,:)),(T850AdvW_jja(:,y,:)),1);  % no need double
  wrk = lsm_jja(:,y);
  krAdvW_y(y,:) = polyfit((T850_jja(wrk,y,:)),(T850AdvW_jja(wrk,y,:)),1);  % no need double
  kcAdvW_y(y,:) = polyfit((T850tend_jja(wrk,y,:)),(T850AdvW_jja(wrk,y,:)),1);  % no need double
end
krAdvW_y = krAdvW_y(:,1)*86400;
kcAdvW_y = kcAdvW_y(:,1);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,1,1)
plot(lat0,krAdvU_y,'c-o',lat0,krAdvV_y,'b-v',lat0,krAdvW_y,'r-o',lat0,krErr_y,'g-o')
legend('U Adv','V Adv','W Adv','Error');
xlabel('Latitude (deg)'); ylabel('regress to T anomaly (day^{-1})'); grid on
title('JJA');

subplot(2,1,2)
plot(lat0,kcAdvU_y,'c-o',lat0,kcAdvV_y,'b-v',lat0,kcAdvW_y,'r-o',lat0,kcErr_y,'g-o')
legend('U Adv','V Adv','W Adv','Error');
xlabel('Latitude (deg)'); ylabel('regress to T tendency (unitless)'); grid on

savefig(gcf,['TendencyRegressJJA_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['TendencyRegressJJA_',textTH,'_',text,'.pdf'])

% DJF
krErr_y = nan([ds(2) 2]);  % so-called real component
kcErr_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krErr_y(y,:) = polyfit((T850_djf(:,y,:)),(T850err_djf(:,y,:)),1);  % no need double
%  kcErr_y(y,:) = polyfit((T850tend_djf(:,y,:)),(T850err_djf(:,y,:)),1);  % no need double
  wrk = lsm_djf(:,y);
  krErr_y(y,:) = polyfit((T850_djf(wrk,y,:)),(T850err_djf(wrk,y,:)),1);  % no need double
  kcErr_y(y,:) = polyfit((T850tend_djf(wrk,y,:)),(T850err_djf(wrk,y,:)),1);  % no need double
end
krErr_y = krErr_y(:,1)*86400;
kcErr_y = kcErr_y(:,1);

krAdvU_y = nan([ds(2) 2]);  % so-called real component
kcAdvU_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krAdvU_y(y,:) = polyfit((T850_djf(:,y,:)),(T850AdvU_djf(:,y,:)),1);  % no need double
%  kcAdvU_y(y,:) = polyfit((T850tend_djf(:,y,:)),(T850AdvU_djf(:,y,:)),1);  % no need double
  wrk = lsm_djf(:,y);
  krAdvU_y(y,:) = polyfit((T850_djf(wrk,y,:)),(T850AdvU_djf(wrk,y,:)),1);  % no need double
  kcAdvU_y(y,:) = polyfit((T850tend_djf(wrk,y,:)),(T850AdvU_djf(wrk,y,:)),1);  % no need double
end
krAdvU_y = krAdvU_y(:,1)*86400;
kcAdvU_y = kcAdvU_y(:,1);

krAdvV_y = nan([ds(2) 2]);  % so-called real component
kcAdvV_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krAdvV_y(y,:) = polyfit((T850_djf(:,y,:)),(T850AdvV_djf(:,y,:)),1);  % no need double
%  kcAdvV_y(y,:) = polyfit((T850tend_djf(:,y,:)),(T850AdvV_djf(:,y,:)),1);  % no need double
  wrk = lsm_djf(:,y);
  krAdvV_y(y,:) = polyfit((T850_djf(wrk,y,:)),(T850AdvV_djf(wrk,y,:)),1);  % no need double
  kcAdvV_y(y,:) = polyfit((T850tend_djf(wrk,y,:)),(T850AdvV_djf(wrk,y,:)),1);  % no need double
end
krAdvV_y = krAdvV_y(:,1)*86400;
kcAdvV_y = kcAdvV_y(:,1);

krAdvW_y = nan([ds(2) 2]);  % so-called real component
kcAdvW_y = nan([ds(2) 2]);  % so-called complex component
for y = 33:ds(2)
%  krAdvW_y(y,:) = polyfit((T850_djf(:,y,:)),(T850AdvW_djf(:,y,:)),1);  % no need double
%  kcAdvW_y(y,:) = polyfit((T850tend_djf(:,y,:)),(T850AdvW_djf(:,y,:)),1);  % no need double
  wrk = lsm_djf(:,y);
  krAdvW_y(y,:) = polyfit((T850_djf(wrk,y,:)),(T850AdvW_djf(wrk,y,:)),1);  % no need double
  kcAdvW_y(y,:) = polyfit((T850tend_djf(wrk,y,:)),(T850AdvW_djf(wrk,y,:)),1);  % no need double
end
krAdvW_y = krAdvW_y(:,1)*86400;
kcAdvW_y = kcAdvW_y(:,1);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,1,1)
plot(lat0,krAdvU_y,'c-o',lat0,krAdvV_y,'b-v',lat0,krAdvW_y,'r-o',lat0,krErr_y,'g-o')
legend('U Adv','V Adv','W Adv','Error');
xlabel('Latitude (deg)'); ylabel('regress to T anomaly (day^{-1})'); grid on
title('DJF');

subplot(2,1,2)
plot(lat0,kcAdvU_y,'c-o',lat0,kcAdvV_y,'b-v',lat0,kcAdvW_y,'r-o',lat0,kcErr_y,'g-o')
legend('U Adv','V Adv','W Adv','Error');
xlabel('Latitude (deg)'); ylabel('regress to T tendency (unitless)'); grid on

savefig(gcf,['TendencyRegressDJF_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['TendencyRegressDJF_',textTH,'_',text,'.pdf'])
%}

%% composite vertical profile 20180127
%{
clear;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
caseid=['ERA-interim_19790101-20151231'];
verX='x903';
load(['temp_',verX,'_',caseid,'.mat'],'caseid','strTitleX','thresh' ,'ds','yStart','yEnd','nyr','f_h2d','time','latt42','lont42' ,'ds_jja','nd_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend','lsm_jja' );
%whos; disp('pause'); pause;
disp('finish load'); toc

%% load nc, check lat lon
nc_z = ['../ERA-interim/nco_z_00Z_',caseid,'.nc'];
Z = single(ncread(nc_z,'z'));
lattmp = ncread(nc_z,'latitude');
lontmp = ncread(nc_z,'longitude');
if (max(abs(latt42-lattmp))>0.1)
  Z = Z(:,end:-1:1 ,:,:);
  lattmp = lattmp(end:-1:1);
end
if (max(abs(latt42-lattmp))>0.1 || max(abs(lont42-lontmp))>0.1)
  disp('error')
end

nc_s500 = ['../ncl/daily500_',caseid,'.nc'];
S500 = single(ncread(nc_s500,'S500Daily'));
lattmp = ncread(nc_s500,'latitude');
lontmp = ncread(nc_s500,'longitude');
if (max(abs(latt42-lattmp))>0.1)
  S500 = S500(:,end:-1:1 ,:);
  lattmp = lattmp(end:-1:1);
end
if (max(abs(latt42-lattmp))>0.1 || max(abs(lont42-lontmp))>0.1)
  disp('error')
end
disp('finish ncload'); toc

%% x9xx: mimic SKSanomaly.f90
Z_star = movmean(Z, 15,4);  % no Endpoints treatment for star..
Z_star = movmean(Z_star, 15,4);  % no Endpoints treatment for star..
Z_hat = nan([ds(1:2) 10 366 nyr],'single');
S500_star = movmean(S500, 15,3);  % no Endpoints treatment for star..
S500_star = movmean(S500_star, 15,3);  % no Endpoints treatment for star..
S500_hat = nan([ds(1:2) 366 nyr],'single');

for t = 1:366
  tArr = days( datetime('0000-01-01')+caldays(t-1)+calyears(yStart:yEnd) - f_h2d(time(1)) )+1; % order of addition matters
  Z_hat(:,:,:,t,:) = Z_star(:,:,:,tArr);
  S500_hat(:,:,t,:) = S500_star(:,:,tArr);
end

Z_hat(:,:,:,[1:104,end-75:end]) = nan;  % jump elsewhere not Jan 1..
Z_hat = movmean(Z_hat,11,5,'omitnan');
S500_hat(:,:,[1:104,end-75:end]) = nan;  % jump elsewhere not Jan 1..
S500_hat = movmean(S500_hat,11,4,'omitnan');

dtArr = f_h2d(time); DArr = 366*(dtArr.Year-yStart);
dtArr.Year=0; DArr = DArr + days(dtArr - datetime('0000-01-01') )+1;
Z = Z - Z_hat(:,:,:,DArr);  % prime
S500 = S500 - S500_hat(:,:,DArr);  % prime
%Z(isnan(Z)) = 0;
%ZCli = Z_hat(:,:,:,DArr);
clear Z_bar Z_hat tArr dtArr DArr  %Z_star
clear ZCli Z_star  % QC
clear S500_bar S500_hat  S500_star S500Cli

%% collect JJA
Z_jja = zeros([ds(1:2) 10 ds_jja(3)],'single');
S500_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  Z_jja(:,:,:,tpointer+(0:tend-tstart)) = Z(:,:,:,tstart:tend);
  S500_jja(:,:,tpointer+(0:tend-tstart)) = S500(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
%{
Z_djf = zeros([ds(1:2) 10 ds_djf(3)],'single');
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  Z_djf(:,:,:,tpointer+(0:tend-tstart)) = Z(:,:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
%}
clear Z S500

disp('finish preproc'); toc

%composite
%
Z_jja = reshape(permute(Z_jja,[1 2 4 3]),[prod(ds_jja) 10]);  %sub2ind

HotZ_z = mean(Z_jja(Hot_jja,:),1);
HotT_z = -diff(HotZ_z)./log([2:10]./[1:9])/287*9.81;
HotS_z = -diff(HotT_z)./100 + 2/7*(HotT_z(2:end)+HotT_z(1:end-1))/2./[200:100:900];  % K/hPa
HotS500 = mean(S500_jja(Hot_jja))*100;

fn_figure = ['pdfST_',verX,'lsm.ps'];
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,3,1);
plot( HotZ_z, 100:100:1000,'-o');
xlabel('Z anomaly (m)');
axis ij; grid on;
subplot(1,3,2);
plot( HotT_z, 150:100:950,'-o');
xlabel('T anomaly (K)');
axis ij; grid on;
subplot(1,3,3);
plot( HotS_z, 200:100:900,'-o',  HotS500,500,'x');
xlabel('S anomaly (K/hPa)');
axis ij; grid on; ylim([100 1000]);
%print(gcf, '-dpsc2','-append',fn_figure);
%}

% sort index 2018
%{
[~,indz] = sort(Z_jja,4); [~,indz] = sort(indz,4); indz=int16(indz);
[~,indx] = sort(mx2t_jja,3); [~,indx] = sort(indx,3); indx=int16(indx);
[~,inds] = sort(S500_jja,3); [~,inds] = sort(inds,3); inds=int16(inds);
clear Z_jja mx2t_jja Hot_jja S500_jja
disp('finish sort'); toc

indz = reshape(indz,[prod(ds(1:2)) 10 ds_jja(3)]);
indx = reshape(indx,[prod(ds(1:2)) ds_jja(3)]);
inds = reshape(inds,[prod(ds(1:2)) ds_jja(3)]);

nbins=50; edges=[0:nbins]/nbins; midpt=(edges(1:end-1)+edges(2:end))/2;
%indz2 = discretize(indz, ds_jja(3)*edges,'categorical');
%indx2 = discretize(indx, ds_jja(3)*edges,'categorical');

%hist3
pdf2 = zeros([nbins nbins 10]);
for k=1:10
%  pdf2 = histcounts2(indx,squeeze(indz(:,:,5,:)), ds_jja(3)*edges,ds_jja(3)*edges);
  pdf2(:,:,k) = histcounts2(indx(lsm_jja,:),squeeze(indz(lsm_jja,k,:)), ds_jja(3)*edges,ds_jja(3)*edges);
end
pdf2 = pdf2/nnz(lsm_jja)/ds_jja(3)*nbins^2;
rankavg = mean(pdf2.*repmat(midpt,[nbins 1 10]),2)./mean(pdf2,2);

%xxx = indx(lsm_jja,:);
%yyy = xxx+34*5;%max(squeeze(indz(lsm_jja,5,:)), xxx);
%pdf2 = histcounts2(xxx,yyy, ds_jja(3)*edges,ds_jja(3)*edges);
%pdf2 = pdf2/nnz(lsm_jja)/ds_jja(3)*nbins^2;
%rankavg = mean(pdf2.*repmat(midpt,[nbins 1]),2);

addpath('/n/home05/pchan/bin');
fn_figure = ['pdfZT_',verX,'lsm.ps'];
%system(['rm ',fn_figure]);
for k=1:10
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(midpt, midpt,pdf2(:,:,k)');
caxis([0 5]); colorbar; caxis auto;
colormap(gca,flipud(hot));
%colormap(gca,b2r(-1,1));
hold on; plot(midpt,rankavg(:,k),'linewidth',2);
  title({strTitleX,['Z',num2str(k*100)]},'FontSize',20);
xlabel('Rank of surface temperature'); ylabel('Rank of geopotential height');
axis square; %axis ij;
set(gca,'FontSize',14);
%print(gcf, '-dpsc2','-append',fn_figure);
end
%system(['ps2pdf ',fn_figure]);

%ocean >40N
ocn_jja = ~lsm_jja; ocn_jja(:,latt42(:)<=thresh{5})=false;
pdf2 = zeros([nbins nbins 10]);
for k=1:10
  pdf2(:,:,k) = histcounts2(indx(ocn_jja,:),squeeze(indz(ocn_jja,k,:)), ds_jja(3)*edges,ds_jja(3)*edges);
end
pdf2 = pdf2/nnz(ocn_jja)/ds_jja(3)*nbins^2;
rankavg = mean(pdf2.*repmat(midpt,[nbins 1 10]),2)./mean(pdf2,2);

addpath('/n/home05/pchan/bin');
fn_figure = ['pdfZT_',verX,'ocn.ps'];
system(['rm ',fn_figure]);
for k=1:10
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(midpt, midpt,pdf2(:,:,k)');
caxis([0 5]); colorbar; caxis auto;
colormap(gca,flipud(hot));
%colormap(gca,b2r(-1,1));
hold on; plot(midpt,rankavg(:,k),'linewidth',2);
%  title({strTitleX,['Z',num2str(k*100)]},'FontSize',20);
xlabel('Rank of surface temperature'); ylabel(['Rank of Z',num2str(k*100)]);
axis square; %axis ij;
set(gca,'FontSize',14);
print(gcf, '-dpsc2','-append',fn_figure);
end
system(['ps2pdf ',fn_figure]);

%S500
pdf2 = histcounts2(indx(lsm_jja,:),inds(lsm_jja,:), ds_jja(3)*edges,ds_jja(3)*edges);
pdf2 = pdf2/nnz(lsm_jja)/ds_jja(3)*nbins^2;
rankavg = mean(pdf2.*repmat(midpt,[nbins 1]),2)./mean(pdf2,2);

addpath('/n/home05/pchan/bin');
fn_figure = ['pdfST_',verX,'lsm.ps'];
system(['rm ',fn_figure]);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(midpt, midpt,pdf2');
caxis([0 1.6]); colorbar; caxis auto;
colormap(gca,flipud(hot));
%colormap(gca,b2r(-1,1));
hold on; plot(midpt,rankavg,'linewidth',2);
%  title({strTitleX,['Z',num2str(k*100)]},'FontSize',20);
xlabel('Rank of surface temperature'); ylabel('Rank of static stab. 500hPa');
axis square; %axis ij;
set(gca,'FontSize',14);
print(gcf, '-dpsc2','-append',fn_figure);
%system(['ps2pdf ',fn_figure]);
%}

%

%% polyfit / plot
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'])
%

thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

%textTH = [textTH, '_chunk'];
%textTH = [textTH, '_chunkwgt'];
%textTH = [textTH, '_org'];
textTH = [textTH, '_wgt'];

% colocate
%{
HotHemi = sum(Hot_jja ,3);
PERfreq0202 = mean(PER0202_jja,3);
PERfreq2225 = mean(PER2225_jja,3);
PERfreq4001 = mean(PER4001_jja,3);
%PERfreq5000 = mean(PER5000,3);
%PERfreq5001 = mean(PER5001,3);

HotPod0202 = sum(Hot_jja &PER0202_jja, 3)./HotHemi;
HotPod2225 = sum(Hot_jja &PER2225_jja, 3)./HotHemi;
HotPod4001 = sum(Hot_jja &PER4001_jja, 3)./HotHemi;
%HotPod5000 = sum(Hot_jja &PER5000, 3)./HotHemi;
%HotPod5001 = sum(Hot_jja &PER5001, 3)./HotHemi;
%}

    lat_bnds = ncread(fn_t42,'lat_bnds');
    R = 6371; %km
%    areaEarth = (2*pi)*(R^2)* (cosd(lat1a') *1*pi/180);
    areaEarth = (2*pi)*(R^2)* (sind(lat_bnds(2,:))-sind(lat_bnds(1,:)));
%    areaEarth = areaEarth(1:floor(ds_jja(2)/2));
  areaEarth(lat0(:)<=thresh{5})=0;

if (contains(textTH, 'wgt'))
  Hot_yt  = squeeze(mean(Hot_jja.*(mx2t_jja),1));
  Cold_yt  = squeeze(mean(Cold_djf.*(mn2t_djf),1));
else
  Hot_yt  = squeeze(mean(Hot_jja,1));
  Cold_yt  = squeeze(mean(Cold_djf,1));
end

ver=['0602_',caseid];
if (contains(textTH, 'wgt'))
  PERjja_yt  = squeeze(mean(PER0202_jja.*(Z500a_jja),1));
  PERdjf_yt  = squeeze(mean(PER0202_djf.*(Z500a_djf),1));
else
  PERjja_yt  = squeeze(mean(PER0202_jja,1));
  PERdjf_yt  = squeeze(mean(PER0202_djf,1));
end

%nyr=21; ver=['5000_',caseid];  %asdfa
%PERjja_yt  = squeeze(mean(PER5000,1));
%nyr=21; ver=['5001_',caseid];  %asdfa
%PERjja_yt  = squeeze(mean(PER5001,1));

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
%plot(bjjaArea_t(:),hotArea_t(:),'.','markersize',3); %%%%%%%%%%

 fn_save  = ['scatter_',textTH,'_',ver,'.mat'];
 save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');

ver=['2631_',caseid]; u=2;v=1;  %Hybrid2020, zprime02, bugfix+4
if (contains(textTH, 'wgt'))
  PERjja_yt  = squeeze(mean(PER2225_jja.*(Z500a_jja),1));
  PERdjf_yt  = squeeze(mean(PER2225_djf.*(Z500a_djf),1));
else
  PERjja_yt  = squeeze(mean(PER2225_jja,1));
  PERdjf_yt  = squeeze(mean(PER2225_djf,1));
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

 fn_save  = ['scatter_',textTH,'_',ver,'.mat'];
 save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');

ver=['4001_',caseid];  %NewReversal, bugfix+1
if (contains(textTH, 'wgt'))
  PERjja_yt  = squeeze(mean(PER4001_jja.*(Z500a_jja),1));
  PERdjf_yt  = squeeze(mean(PER4001_djf.*(Z500a_djf),1));
else
  PERjja_yt  = squeeze(mean(PER4001_jja,1));
  PERdjf_yt  = squeeze(mean(PER4001_djf,1));
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

 fn_save  = ['scatter_',textTH,'_',ver,'.mat'];
 save(fn_save,'bjjaArea_t','hotArea_t','hotstat','bdjfArea_t','coldArea_t','coldstat');
%}

% plot
%{
hFig=figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);

thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd')

thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];
%textTH = [textTH, '_chunk'];
textTH = [textTH, '_chunkwgt'];

ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(2,4,1)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
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
  contour( Xedges(3:end),Yedges,Ncounts(3:end,:).^0.50','k');
  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.50',3,'k');
end
title({'Dunn-Sigouin and Son 2013 index', ['hot, r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bjjaArea_t(:)),max(bjjaArea_t(:)), min(hotArea_t(:)),max(hotArea_t(:))]);

subplot(2,4,5)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
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
  contour( Xedges(2:end),Yedges,Ncounts(2:end,:).^0.40','k');
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title(['cold, r=',num2str(coldstat(3),'%+.3f')])
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))]);

ver=['2631_',caseid]; u=2;v=1;  %Hybrid2020, zprime02, bugfix+4
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(2,4,2)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
plot(bjjaArea_t(:),hotArea_t(:),'.','markersize',3);
plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))],[min(bjjaArea_t(:)),max(bjjaArea_t(:))]*hotstat(1)+hotstat(2),'-','linewidth',2)
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
  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)))]);
  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bjjaArea_t(:),hotArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(2:end),Yedges,Ncounts(2:end,:).^0.40',1,'k');
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.30',3,'k');
end
title({'Hassanzadeh et al. 2014 index', ['hot, r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bjjaArea_t(:)),max(bjjaArea_t(:)), min(hotArea_t(:)),max(hotArea_t(:))]);

subplot(2,4,6)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
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
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bdjfArea_t(:),coldArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.30','k');
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.30',3,'k');
end
title(['cold, r=',num2str(coldstat(3),'%+.3f')])
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))]);

ver=['4001_',caseid];  %NewReversal, bugfix+1
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(2,4,3)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
plot(bjjaArea_t(:),hotArea_t(:),'.','markersize',3);
plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))],[min(bjjaArea_t(:)),max(bjjaArea_t(:))]*hotstat(1)+hotstat(2),'-','linewidth',2)
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
  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)))]);
  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bjjaArea_t(:),hotArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({'Scherrer et al. 2006 index', ['hot, r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bjjaArea_t(:)),max(bjjaArea_t(:)), min(hotArea_t(:)),max(hotArea_t(:))]);

subplot(2,4,7)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
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
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bdjfArea_t(:),coldArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title(['cold, r=',num2str(coldstat(3),'%+.3f')])
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))]);

%{
ver=['7050_',caseid];  %
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(2,4,4)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
plot(bjjaArea_t(:),hotArea_t(:),'.','markersize',3);
plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))],[min(bjjaArea_t(:)),max(bjjaArea_t(:))]*hotstat(1)+hotstat(2),'-','linewidth',2)

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart:yEnd
    text(double(bjjaArea_t(yyyy-yStart+1)),double(hotArea_t(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)))]);
  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bjjaArea_t(:),hotArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({'v850 quantile (7050)', ['hot, r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bjjaArea_t(:)),max(bjjaArea_t(:)), min(hotArea_t(:)),max(hotArea_t(:))]);

subplot(2,4,8)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
plot(bdjfArea_t(:),coldArea_t(:),'.','markersize',3);
plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))],[min(bdjfArea_t(:)),max(bdjfArea_t(:))]*coldstat(1)+coldstat(2),'-','linewidth',2)

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart+1:yEnd
    text(double(bdjfArea_t(yyyy-yStart)),double(coldArea_t(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bdjfArea_t(:),coldArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title(['cold, r=',num2str(coldstat(3),'%+.3f')])
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))]);
%}

savefig(gcf,['scatter_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['scatter_',textTH,'_',text,'.pdf'])
%}


%% save append
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textWgtArr = {'chunk'};
for textWgt = textWgtArr

textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

%{
ver=['0602_',caseid];
  strTitle = 'Dunn-Sigouin and Son 2013 index';
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);

ver=['2631_',caseid];
  strTitle = 'Hassanzadeh et al. 2014 index';
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);

ver=['4002_',caseid];
  strTitle = 'Scherrer et al. 2006 index';
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);

ver=['5600_',caseid];
  strTitle = 'Pfahl and Wernli 2012 index';
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);

ver=['8000_',caseid];
  strTitle = 'Masato et al. 2013b index';
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);
%}

%
ver=['4002_',caseid];
  strTitle = ['Scherrer et al. 2006 index',10,'21-74N, with second criterion'];
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);

ver=['4004_',caseid];
  strTitle = ['Scherrer et al. 2006 index',10,'46-74N, with second criterion'];
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);

ver=['4014_',caseid];
  strTitle = ['Scherrer et al. 2006 index',10,'46-74N, without second criterion'];
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);
%

ver=['x600_',caseid];
  strTitle = ['Extreme Cold Area'];
  save(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle','-append');
  whos('-file', ['../index_wise/scatter_',textTH,'_',ver,'.mat']);

end % textWgt loop
%}

%% automated plot polyfit
%{
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd');

fn_figure = ['panel_',textTH,'_',text,'.ps'];
system(['rm ',fn_figure]);  % TODO

%vernumArr = {'5600','4002','8000','0602','2631'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633'};
%vernumArr = {'5900','4923','8900','0902','2933'};
%vernumArr = {'5900','4923','8900','0a02','2937'};  % 170621
vernumArr = {'5940','4943','2946','x900'};  % cutoff low
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
%textWgtArr = {'chunk','chunkwgt'};
textWgtArr = {'chunk'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];
%textTH = [textTH, '_chunk'];
%textTH = [textTH, '_chunkwgt'];

% Hot
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)-1  %TODO

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
  load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle');
subplot(2,3,m);
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
plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))],[min(bjjaArea_t(:)),max(bjjaArea_t(:))]*hotstat(1)+hotstat(2),'-','linewidth',2);
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
%  disp(['hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',hotArea_t(:)))]);
%  system(['echo hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',hotArea_t(:))), ' >> corrtrend-hot']);
%  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)))]);
%  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bjjaArea_t(:),hotArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({strTitle, ['hot, r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bjjaArea_t(:)),max(bjjaArea_t(:)), min(hotArea_t(:)),max(hotArea_t(:))]); axis square; %axis tight;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);

% Cold
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)-1  %TODO

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
  load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle');
subplot(2,3,m);
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
plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))],[min(bdjfArea_t(:)),max(bdjfArea_t(:))]*coldstat(1)+coldstat(2),'-','linewidth',2);
% http://www.originlab.com/doc/Origin-Help/LR-Algorithm   confidence ellipse
%inv22 = inv( cov(bdjfArea_t(:),coldArea_t(:)) );  % normalize by N-1
%xmean=mean(bdjfArea_t(:)); ymean=mean(coldArea_t(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))],'r','linewidth',2,'LevelList',1);
%plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))], ([min(bdjfArea_t(:)),max(bdjfArea_t(:))]-xmean)*tan(atan2(-2*inv22(2),inv22(4)-inv22(1))/2)+ymean,'-','linewidth',2)

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart+1:yEnd
    text(double(bdjfArea_t(yyyy-yStart)),double(coldArea_t(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
%  disp(['cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',coldArea_t(:)))]);
%  system(['echo cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',coldArea_t(:))), ' >> corrtrend-cold']);
%  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
%  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bdjfArea_t(:),coldArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({strTitle, ['cold, r=',num2str(coldstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))]); axis square; %axis tight;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);

end % textWgt loop

%system(['ps2pdf ',fn_figure]);  % TODO
%

%% automated Block Freq (Pfahl2a in xtrm_colocate_pchan)
%
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd','latt42','lont42');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panel_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);

%vernumArr = {'5600','4002','8000','0602','2631','x600'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
%textWgtArr = {'chunk','chunkwgt'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

% JJA
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
subplot(2,3,m);

if (contains(textTH, 'wgt'))
  PERfreq_jja = mean(PER_jja.*(Wgt_jja),3);
else
  PERfreq_jja = mean(PER_jja,3);
end
%PERfreq_jja(PERfreq_jja==0) = nan;
if (isfield(prm,'yN1'))
  PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
else
  PERfreq_jja(:,[1:end/2]) = nan;
end

axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_jja'); shading flat;
colormap(jet(10)); caxis([0 10]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'k');
%title('\fontsize{20}Relative frequency (%) of intense blocking events during JJA');
if (contains(textTH, 'wgt'))
  title({strTitle,'JJA weighted blocking frequency (unit*%)'},'fontsize',11);
else
  title({strTitle,'JJA blocking frequency (%)'},'fontsize',11);
end
tightmap;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);

% DJF
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
subplot(2,3,m);

if (contains(textTH, 'wgt'))
  PERfreq_djf = mean(PER_djf.*(Wgt_djf),3);
else
  PERfreq_djf = mean(PER_djf,3);
end
%PERfreq_djf(PERfreq_djf==0) = nan;
if (isfield(prm,'yN1'))
  PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
else
  PERfreq_djf(:,[1:end/2]) = nan;
end

axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(jet(10)); caxis([0 10]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'k');
%title('\fontsize{20}Relative frequency (%) of intense blocking events during DJF');
if (contains(textTH, 'wgt'))
  title({strTitle,'DJF weighted blocking frequency (unit*%)'},'fontsize',11);
else
  title({strTitle,'DJF blocking frequency (%)'},'fontsize',11);
end
tightmap;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);

end % textWgt loop

%system(['ps2pdf ',fn_figure]);  % TODO
%}

%% automated time series (aka legend) & table of correlation
%{
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd');

fn_figure = ['panel_',textTH,'_',text,'.ps'];
system(['rm ',fn_figure]);  % TODO

%vernumArr = {'5600','4002','8000','0602','2631'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%vernumArr = {'5900','5940','4923','4933','4934','4943','8900','0a02','2937','2946','x900'};  % 170622
%vernumArr = {'5900','0a02','2937','4923','8900','5940','2946','4943','x900'};  % 170623
vernumArr = {'5900','0a02','2937','4923','8900','5940','2946','4943'};  % 170623
textWgtArr = {'chunk'};
%textWgtArr = {'chunk','chunkwgt'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

strLegend = cell(length(vernumArr),1);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
  load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle');

bjjaArr(:,m) = bjjaArea_t(:);
bdjfArr(:,m) = bdjfArea_t(:);
if (contains(textTH, 'wgt'))
  bjjaArr(:,m) = bjjaArr(:,m) / mean(bjjaArr(:,m));
  bdjfArr(:,m) = bdjfArr(:,m) / mean(bdjfArr(:,m));
end
strLegend{m} = strTitle;
  end  % m
%strLegend = {'P12','S06','M13','D13','H14','Extreme'};
%strLegend = {'P12','D13','H14','S06','M13','Modified P12','Modified H14','Modified S06','Extreme'};  %TODO
strLegend = {'P12','D13','H14','S06','M13','Modified P12','Modified H14','Modified S06'};  %TODO

%{
figure('units','inches','position',[0 1 16 8], 'paperUnits','inches','papersize',[16 8],'paperposition',[0 0 16 8]);
% Hot
subplot(2,1,1);
hold on;

if (contains(textTH, 'wgt'))
  xlabel({'year'}); title({'Mean weighted JJA blocking area (normalized)'});
else
  xlabel({'year'}); title({'Mean JJA blocking area (km^2)'});
end
%plot([yStart:yEnd]', bjjaArr);
plot([yStart:yEnd]', bjjaArr(:,1:end-1));
plot([yStart:yEnd]', bjjaArr(:,end),'k');

ax=gca; ax.ColorOrderIndex = 1;
  for m = 1:length(vernumArr)-1  %TODO
    tmpstat = [polyfit((yStart:yEnd)',bjjaArr(:,m),1)];
    fplot(@(x) polyval(tmpstat,x), [yStart yEnd], '--');
%    title(['Legend, r=',num2str(corr((yStart:yEnd)',bjjaArea_t(:)'),'%+.3f')]);
  end  % m
  tmpstat = [polyfit((yStart:yEnd)',bjjaArr(:,end),1)];
  fplot(@(x) polyval(tmpstat,x), [yStart yEnd], 'k--');
xlim([yStart yEnd]); %axis square;
legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

% Cold
subplot(2,1,2);
hold on;

if (contains(textTH, 'wgt'))
  xlabel({'year'}); title({'Mean weighted DJF blocking area (normalized)'});
else
  xlabel({'year'}); title({'Mean DJF blocking area (km^2)'});
end
%plot([yStart+1:yEnd]', bdjfArr);
plot([yStart+1:yEnd]', bdjfArr(:,1:end-1));
plot([yStart+1:yEnd]', bdjfArr(:,end),'k');

ax=gca; ax.ColorOrderIndex = 1;
  for m = 1:length(vernumArr)-1  %TODO
    tmpstat = [polyfit((yStart+1:yEnd)',bdjfArr(:,m),1)];
    fplot(@(x) polyval(tmpstat,x), [yStart yEnd], '--');
%    title(['Legend, r=',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)'),'%+.3f')]);
  end  % m
  tmpstat = [polyfit((yStart+1:yEnd)',bdjfArr(:,end),1)];
  fplot(@(x) polyval(tmpstat,x), [yStart yEnd], 'k--');
xlim([yStart+1 yEnd]); %axis square;
legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

print(gcf, '-dpsc2','-append',fn_figure);
%}

%% table of correlation
rhojja = corr(bjjaArr);
disp(rhojja)
rhojja(repmat([1:length(vernumArr)]',[1 length(vernumArr)])<=repmat([1:length(vernumArr)],[length(vernumArr) 1])) = nan;
rhodjf = corr(bdjfArr);
disp(rhodjf)
rhodjf(repmat([1:length(vernumArr)]',[1 length(vernumArr)])<=repmat([1:length(vernumArr)],[length(vernumArr) 1])) = nan;

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 16 8], 'paperUnits','inches','papersize',[16 8],'paperposition',[0 0 16 8]);
% JJA
subplot(1,2,1);
pcolorPH(1:length(vernumArr),1:length(vernumArr), rhojja);
colormap(gca,b2r(-1,1)); %colorbar;
  mytext = text;
  clear text;
  for m = 1:length(vernumArr)
  for n = m+1:length(vernumArr)
    text(double(m),double(n), sprintf('%+.1f',rhojja(n,m)) ,'HorizontalAlignment','center','fontsize',16);
  end
  end
  text = mytext;
title('Correlation between different indices: JJA','fontsize',20);
xticks(1:length(vernumArr)); xticklabels(strLegend); xtickangle(45);
yticks(1:length(vernumArr)); yticklabels(strLegend);
axis square; axis ij;
set(gca,'fontsize',20);

% DJF
subplot(1,2,2);
pcolorPH(1:length(vernumArr),1:length(vernumArr), rhodjf);
colormap(gca,b2r(-1,1)); %colorbar;
  mytext = text;
  clear text;
  for m = 1:length(vernumArr)
  for n = m+1:length(vernumArr)
    text(double(m),double(n), sprintf('%+.1f',rhodjf(n,m)) ,'HorizontalAlignment','center','fontsize',16);
  end
  end
  text = mytext;
title('Correlation between different indices: DJF','fontsize',20);
xticks(1:length(vernumArr)); xticklabels(strLegend); xtickangle(45);
yticks(1:length(vernumArr)); yticklabels(strLegend);
axis square; axis ij;
set(gca,'fontsize',20);

print(gcf, '-dpsc2','-append',fn_figure);


end % textWgt loop

system(['ps2pdf ',fn_figure]);  % TODO
%}

%% automated ENSO polyfit (see fig4b)
%{
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd');

fn_figure = ['panel_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);  % TODO

%vernumArr = {'5600','4002','8000','0602','2631','x600'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
%textWgtArr = {'chunk','chunkwgt'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];
%textTH = [textTH, '_chunk'];
%textTH = [textTH, '_chunkwgt'];

% wget http://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
% awk 'BEGIN{FIELDWIDTHS = "5 5 7 7"} $1=="  JJA" && $2>=1979 && $2<=2015  {print $0}' oni.ascii.txt > oni_JJA.txt
% awk 'BEGIN{FIELDWIDTHS = "5 5 7 7"} $1=="  DJF" && $2>=1980 && $2<=2015  {print $0}' oni.ascii.txt > oni_DJF.txt

% Hot
oni_JJA = dlmread('oni_JJA.txt','',0,1);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
  load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle');
hotstat = [polyfit(oni_JJA(:,3),bjjaArea_t(:),1) corr(oni_JJA(:,3),bjjaArea_t(:))];
subplot(2,3,m);
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',oni_JJA(:,3)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
  xlabel({'Oceanic Ni$\tilde{n}$o Index',strXCorr},'interpreter','latex');
  ylabel({'Weighted blocking area (unit*km^2)',strYCorr});
else
  xlabel({'Oceanic Ni$\tilde{n}$o Index',strXCorr},'interpreter','latex');
  ylabel({'Blocking area (km^2)',strYCorr});
end
plot(oni_JJA(:,3),bjjaArea_t(:),'.','markersize',3);
plot([min(oni_JJA(:,3)),max(oni_JJA(:,3))],[min(oni_JJA(:,3)),max(oni_JJA(:,3))]*hotstat(1)+hotstat(2),'-','linewidth',2);
% http://www.originlab.com/doc/Origin-Help/LR-Algorithm   confidence ellipse
%inv22 = inv( cov(oni_JJA(:,3),bjjaArea_t(:)) );  % normalize by N-1
%xmean=mean(oni_JJA(:,3)); ymean=mean(bjjaArea_t(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(oni_JJA(:,3)),max(oni_JJA(:,3)), min(bjjaArea_t(:)),max(bjjaArea_t(:))],'r','linewidth',2,'LevelList',1);
%plot([min(oni_JJA(:,3)),max(oni_JJA(:,3))], ([min(oni_JJA(:,3)),max(oni_JJA(:,3))]-xmean)*tan(atan2(-2*inv22(2),inv22(4)-inv22(1))/2)+ymean,'-','linewidth',2)

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart:yEnd
    text(double(oni_JJA(yyyy-yStart+1,3)),double(bjjaArea_t(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:)))]);
%  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',oni_JJA(:,3))), ' >> corrtrend-hot']);
  disp(['ONI trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',oni_JJA(:,3)))]);
%  system(['echo hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',bjjaArea_t(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(oni_JJA(:,3),bjjaArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({strTitle, ['r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(oni_JJA(:,3)),max(oni_JJA(:,3)), min(bjjaArea_t(:)),max(bjjaArea_t(:))]); axis square; %axis tight;

  end  % m
subplot(2,3,m);
if (contains(textTH, 'wgt'))
  ylabel({'Weighted extreme area (K*km^2)',strYCorr});
else
  ylabel({'Extreme area (km^2)',strYCorr});
end
print(gcf, '-dpsc2','-append',fn_figure);

% Cold
oni_DJF = dlmread('oni_DJF.txt','',0,1);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
  load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle');
coldstat = [polyfit(oni_DJF(:,3),bdjfArea_t(:),1) corr(oni_DJF(:,3),bdjfArea_t(:))];
subplot(2,3,m);
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
  xlabel({'Oceanic Ni$\tilde{n}$o Index',strXCorr},'interpreter','latex');
  ylabel({'Weighted blocking area (unit*km^2)',strYCorr});
else
  xlabel({'Oceanic Ni$\tilde{n}$o Index',strXCorr},'interpreter','latex');
  ylabel({'Blocking area (km^2)',strYCorr});
end
plot(oni_DJF(:,3),bdjfArea_t(:),'.','markersize',3);
plot([min(oni_DJF(:,3)),max(oni_DJF(:,3))],[min(oni_DJF(:,3)),max(oni_DJF(:,3))]*coldstat(1)+coldstat(2),'-','linewidth',2);
% http://www.originlab.com/doc/Origin-Help/LR-Algorithm   confidence ellipse
%inv22 = inv( cov(oni_DJF(:,3),bdjfArea_t(:)) );  % normalize by N-1
%xmean=mean(oni_DJF(:,3)); ymean=mean(bdjfArea_t(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(oni_DJF(:,3)),max(oni_DJF(:,3)), min(bdjfArea_t(:)),max(bdjfArea_t(:))],'r','linewidth',2,'LevelList',1);
%plot([min(oni_DJF(:,3)),max(oni_DJF(:,3))], ([min(oni_DJF(:,3)),max(oni_DJF(:,3))]-xmean)*tan(atan2(-2*inv22(2),inv22(4)-inv22(1))/2)+ymean,'-','linewidth',2)

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart+1:yEnd
    text(double(oni_DJF(yyyy-yStart,3)),double(bdjfArea_t(yyyy-yStart)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
%  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3))), ' >> corrtrend-cold']);
  disp(['ONI trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3)))]);
%  system(['echo cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(oni_DJF(:,3),bdjfArea_t(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({strTitle, ['r=',num2str(coldstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(oni_DJF(:,3)),max(oni_DJF(:,3)), min(bdjfArea_t(:)),max(bdjfArea_t(:))]); axis square; %axis tight;

  end  % m
subplot(2,3,m);
if (contains(textTH, 'wgt'))
  ylabel({'Weighted extreme area (K*km^2)',strYCorr});
else
  ylabel({'Extreme area (km^2)',strYCorr});
end
print(gcf, '-dpsc2','-append',fn_figure);

end % textWgt loop

%system(['ps2pdf ',fn_figure]);  % TODO
%

%% automated map regress on ENSO
%
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'ds','yStart','yEnd','nyr','latt42','lont42');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panel_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);

%vernumArr = {'5600','4002','8000','0602','2631','x600'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
%textWgtArr = {'chunk','chunkwgt'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

% JJA
oni_JJA = dlmread('oni_JJA.txt','',0,1);
oni_JJA = oni_JJA(:,3);
oni_JJA = oni_JJA - nanmean(oni_JJA);
oni_JJA = oni_JJA / sumsqr(oni_JJA);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
subplot(2,3,m);

if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
end
PERjja_reg = sum(PERjja_xyn.*repmat(reshape(oni_JJA,[1 1 nyr]),[ds(1:2) 1]),3);
if (isfield(prm,'yN1'))
  PERjja_reg(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
else
  PERjja_reg(:,[1:end/2]) = nan;
end

axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERjja_reg'); shading flat;
%colormap(jet(10)); caxis([0 10]); colorbar; caxis auto;  % TODO
colormap(gca,b2r(-max(abs(100*PERjja_reg(:))),max(abs(100*PERjja_reg(:))))); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k');
if (contains(textTH, 'wgt'))
%  title({strTitle,['JJA blocking regressed on Oceanic Ni$\tilde{n}$o Index'],'unit*%/K'},'fontsize',11,'interpreter','latex');
  title({strTitle,['JJA blocking regressed on Oceanic Nino Index'],'unit*%/K'},'fontsize',11);
else
%  title({strTitle,['JJA blocking regressed on Oceanic Ni$\tilde{n}$o Index'],'%/K'},'fontsize',11,'interpreter','latex');
  title({strTitle,['JJA blocking regressed on Oceanic Nino Index'],'%/K'},'fontsize',11);
end
tightmap;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);

% DJF
oni_DJF = dlmread('oni_DJF.txt','',0,1);
oni_DJF = oni_DJF(:,3);
oni_DJF = oni_DJF - nanmean(oni_DJF);
oni_DJF = oni_DJF / sumsqr(oni_DJF);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
subplot(2,3,m);

if (contains(textTH, 'wgt'))
  PERdjf_xyn = squeeze(mean( reshape(PER_djf.*(Wgt_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
else
  PERdjf_xyn = squeeze(mean( reshape(PER_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
end
PERdjf_reg = sum(PERdjf_xyn.*repmat(reshape(oni_DJF,[1 1 nyr-1]),[ds(1:2) 1]),3);
if (isfield(prm,'yN1'))
  PERdjf_reg(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
else
  PERdjf_reg(:,[1:end/2]) = nan;
end

axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERdjf_reg'); shading flat;
%colormap(jet(10)); caxis([0 10]); colorbar; caxis auto;  % TODO
colormap(gca,b2r(-max(abs(100*PERdjf_reg(:))),max(abs(100*PERdjf_reg(:))))); colorbar; %caxis auto;  %TODO
plotm(coastlat,coastlon,'k');
if (contains(textTH, 'wgt'))
%  title({strTitle,['DJF blocking regressed on Oceanic Ni$\tilde{n}$o Index'],'unit*%/K'},'fontsize',11,'interpreter','latex');
  title({strTitle,['DJF blocking regressed on Oceanic Nino Index'],'unit*%/K'},'fontsize',11);
else
%  title({strTitle,['DJF blocking regressed on Oceanic Ni$\tilde{n}$o Index'],'%/K'},'fontsize',11,'interpreter','latex');
  title({strTitle,['DJF blocking regressed on Oceanic Nino Index'],'%/K'},'fontsize',11);
end
tightmap;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);

end % textWgt loop

system(['ps2pdf ',fn_figure]);  % TODO
%}


%% automated map yyyy
%{
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'ds','yStart','yEnd','nyr','latt42','lont42');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panelyyyy_',textTH,'_',text,'.ps'];
system(['rm ',fn_figure]);

%vernumArr = {'5600','4002','8000','0602','2631','x600'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
textWgtArr = {'chunk'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

% JJA
%
%yyyy = 2010;
for yyyy = yStart:yEnd
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
subplot(2,3,m);

if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
end
if (isfield(prm,'yN1'))
  PERjja_xyn(:,[1:prm.yN1-1,prm.yN2+1:end],:) = nan;
else
  PERjja_xyn(:,[1:end/2],:) = nan;
end

axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERjja_xyn(:,:,yyyy-yStart+1)'); shading flat;
colormap(jet(10)); caxis([0 10]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'k');
if (contains(textTH, 'wgt'))
  title({strTitle,[num2str(yyyy),' JJA weighted blocking frequency (unit*%)']},'fontsize',11);
else
  title({strTitle,[num2str(yyyy),' JJA blocking frequency (%)']},'fontsize',11);
end
tightmap;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);
end  % yyyy
%

% DJF
%yyyy = 2010;
for yyyy = yStart+1:yEnd
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
subplot(2,3,m);

if (contains(textTH, 'wgt'))
  PERdjf_xyn = squeeze(mean( reshape(PER_djf.*(Wgt_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
else
  PERdjf_xyn = squeeze(mean( reshape(PER_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
end
if (isfield(prm,'yN1'))
  PERdjf_xyn(:,[1:prm.yN1-1,prm.yN2+1:end],:) = nan;
else
  PERdjf_xyn(:,[1:end/2],:) = nan;
end

axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERdjf_xyn(:,:,yyyy-yStart)'); shading flat;
colormap(jet(10)); caxis([0 10]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'k');
if (contains(textTH, 'wgt'))
  title({strTitle,[num2str(yyyy),' DJF weighted blocking frequency (unit*%)']},'fontsize',11);
else
  title({strTitle,[num2str(yyyy),' DJF blocking frequency (%)']},'fontsize',11);
end
tightmap;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);
end  % yyyy

end % textWgt loop

system(['ps2pdf ',fn_figure]);  % TODO
%}
%% automated map yyyy 20180112
%
%caseid=['ERA-interim_19790101-20171231'];
%verX='x913';
load(['temp_',verX,'_',caseid,'.mat'],'thresh','ds','yStart','yEnd','nyr','latt42','lont42','nd_jja','nd_djf');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panelyyyy_',verX,'_',caseid,'.ps'];
system(['rm ',fn_figure]);

%vernumArr = {'5600','4002','8000','0602','2631','x600'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%vernumArr = {'2a53','2a64',verX};  % 180112
%vernumArr = {'2a66','2a70','2a67',verX};  % 180125
%titleArr = {'D=5,A=1.5','prst=.9375,A=2.5','D=1,A=0','Extreme'};
%vernumArr = {'8a10','8a11','8a12',verX};  % 180306
%titleArr = {'8a10','8a11','8a12','Extreme'};
%vernumArr = {'2a66','2a70','2a67','4a25','8a13',verX};  % 180306
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','S06,D=1,GHGN=-10','M13,D=1','Extreme'};
%vernumArr = {'2a66','2a70','2a67','0a12','4a26','4a25','8a13','9a00',verX};  % 180404
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','S06,D=1,GHGN=-10','M13,D=1','FALWA','Extreme'};
%lims = [35,35,100,20,50,50,50,100,20];
%vernumArr = {'2a71','2a68','2a69','0a13','4a26','8a14','9a00',verX};  % 180627
%titleArr = {'DG83,prst=.9375,A=2.5','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA','Extreme'};
%lims = [35,35,100,20,50,50,11e9,20];
%vernumArr = {'9a00','9a05',verX};  % 180627
%titleArr = {'FALWA','FALWA with thresh','Extreme'};
%lims = [11e9,5.5e9,20];
%vernumArr = {'2a73','2a68','2a69','0a13','4a26','8a14','9a06',verX};  % 180730
%titleArr = {'DG83,prst=.5,A=1','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA,D=5,A=median','Extreme'};
%lims = [75,50,100,20,50,50,75,75];  %TODO
%vernumArr = {'2a73','2a68','2a69','0a13','4a28','8a14','9a06',verX};  % 180730
%lims = [75,50,100,20,75,50,75,75];  %TODO

textWgtArr = {'chunk','chunkwgt'};
%textWgtArr = {'chunk'};
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];
textTH = textTH{1};
if (contains(textTH, 'wgt'))
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'PERwjja_n');
  ydata = PERwjja_n;
else
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'PERjja_n');
  ydata = PERjja_n;
end
[~,ysort] = sort(ydata,'descend');

% JJA
%
%for yyyy = yStart:yEnd
for lyr = 1:nyr
system(['echo -ne "\r',num2str(lyr),'"']);
yyyy = yStart-1 + ysort(lyr);
figure('units','inches','position',[1 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
%disp('m1');
load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
%disp('m2'); % load 2nd slow

if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  dPERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ).*repmat(reshape(1:nd_jja,[1 1 nd_jja 1]),[ds(1:2) 1 nyr]), 3)) ./PERjja_xyn;  % x,y,yr
  dPERjja_xyn = (nd_jja+1)/2 +(dPERjja_xyn-(nd_jja+1)/2)*(nd_jja-1)./nd_jja./(1-PERjja_xyn);  % transform
end
if (isfield(prm,'yN1'))
  PERjja_xyn(:,[1:prm.yN1-1,prm.yN2+1:end],:) = nan;
else
  PERjja_xyn(:,[1:end/2],:) = nan;
end

f_tp = @(a,b,c) ceil(c/a) +b*mod(c-1,a);
subplot(3,4,m);
%subplot(3,4,f_tp(3,4,m));
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERjja_xyn(:,:,yyyy-yStart+1)'); shading flat;
%colormap(gca,flipud(hot(10))); caxis([0 lims(m)]); colorbar; %caxis auto;  % TODO
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],10)); caxis([0 lims(m)]); colorbar; %caxis auto;  % TODO
if (contains(textTH, 'wgt'))
  caxis auto;
end
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
%if (contains(textTH, 'wgt'))
%  title({strTitle,[num2str(yyyy),' JJA weighted blocking frequency (unit*%)']},'fontsize',11);
%else
%  title({strTitle,[num2str(yyyy),' JJA blocking frequency (%)']},'fontsize',11);
  title({titleArr{m},[num2str(yyyy),' JJA frequency (%)']},'fontsize',11);
%end
tightmap;

%{
%subplot(2,4,4+m);
subplot(3,6,f_tp(3,6,9+m));
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, dPERjja_xyn(:,:,yyyy-yStart+1)'); shading flat;
colormap(gca,jet(10)); caxis([0 nd_jja]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
  title({[num2str(yyyy),' JJA mean date']},'fontsize',11);
tightmap;
%}
%disp('m4');

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);
end  % yyyy
%

end % textWgt loop

system(['ps2pdf ',fn_figure]);  % TODO
system(['rm ',fn_figure]);
%

%% automated plot ployfit 20180322 (+Block Freq)
%
%% automated plot polyfit
%
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
load(['temp_',verX,'_',caseid,'.mat'],'yStart','yEnd');

fn_figure = ['panel_',verX,'_',caseid,'.ps'];
system(['rm ',fn_figure]);  % TODO

%vernumArr = {'5600','4002','8000','0602','2631'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633'};
%vernumArr = {'5900','4923','8900','0902','2933'};
%vernumArr = {'5900','4923','8900','0a02','2937'};  % 170621
%vernumArr = {'5940','4943','2946','x900'};  % cutoff low
%vernumArr = {'2a66','2a70','2a67','4a25','8a13',verX};  % 180306
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','S06,D=1,GHGN=-10','M13,D=1','Extreme'};
%vernumArr = {'2a66','2a70','2a67','0a12','4a26','4a25','8a13','9a00',verX};  % 180404
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','S06,D=1,GHGN=-10','M13,D=1','FALWA','Extreme'};
%vernumArr = {'2a66','2a70','2a67','0a12','4a26','4a25','8a13','9a00','x905','x906','x907','x908'};  % 180405
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','S06,D=1,GHGN=-10','M13,D=1','FALWA','Extreme1.5','Extreme2','Extreme2.5','Extreme3'};
%vernumArr = {'2a71','2a68','2a69','0a13','4a26','8a14','9a05',verX};  % 180627
%titleArr = {'DG83,prst=.9375,A=2.5','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA,D=5,A=median','Extreme'};
%vernumArr = {'2a72','2a68','2a69','0a13','4a26','8a14','9a06',verX};  % 180730
%titleArr = {'DG83,prst=.97,A=3','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA,D=5,A=median','Extreme'};
%vernumArr = {'2a73','2a68','2a69','0a13','4a26','8a14','9a06',verX};  % 180809
%titleArr = {'DG83,prst=.5,A=1','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA,D=5,A=median','Extreme'};
 %lims = [35,35,100,35,50,50,50,100,20];

%textWgtArr = {'chunk','lndchunk','chunkwgt','lndchunkwgt'};
textWgtArr = {'chunk','lndchunk'};
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];
textTH = textTH{1};

% Hot
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
%ver = [vernumArr{end},'_',caseid];
if (contains(textTH, 'wgt'))
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','PERwjja_n');
  ydata = PERwjja_n;
else
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','PERjja_n');
  ydata = PERjja_n;
end
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rng default;
e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
e99 = (squeeze(mean(e99arr,1)));

  for m = 1:length(vernumArr)-1  %TODO

ver = [vernumArr{m},'_',caseid];
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
%if (contains(textTH, 'wgt'))
%  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERwjja_n');
%  xdata = PERwjja_n;
%else
%  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERjja_n');
%  xdata = PERjja_n;
%end
%if (strcmp(textTH, 'chunk'))
switch textTH
 case 'chunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERjja_n');
  xdata = PERjja_n;
  strT = 'JJA blocking area (km^2)';
 case 'lndchunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERljja_n');
  xdata = PERljja_n;
  strT = 'JJA blocking area on land (km^2)';
 case 'chunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERwjja_n');
  xdata = PERwjja_n;
  strT = 'Weighted JJA blocking area (unit*km^2)';
 case 'lndchunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERlwjja_n');
  xdata = PERlwjja_n;
  strT = 'Weighted JJA blocking area on land (unit*km^2)';
 otherwise
  warning('unexpected textth')
end

subplot(3,4,m);  %TODO
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',xdata(:)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',ydata(:)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
%  xlabel({'Weighted blocking area (unit*km^2)',strXCorr});
  ylabel({'Weighted extreme area (K*km^2)',strYCorr});
else
%  xlabel({'Blocking area (km^2)',strXCorr});
  ylabel({'Extreme area (km^2)',strYCorr});
end
xlabel({strT,strXCorr});
%plot(xdata(:),ydata(:),'.','markersize',3);
%plot([min(xdata(:)),max(xdata(:))],[min(xdata(:)),max(xdata(:))]*hotstat(1)+hotstat(2),'-','linewidth',2);
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
% http://www.originlab.com/doc/Origin-Help/LR-Algorithm   confidence ellipse
%inv22 = inv( cov(xdata(:),ydata(:)) );  % normalize by N-1
%xmean=mean(xdata(:)); ymean=mean(ydata(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(xdata(:)),max(xdata(:)), min(ydata(:)),max(ydata(:))],'r','linewidth',2,'LevelList',1);
%plot([min(xdata(:)),max(xdata(:))], ([min(xdata(:)),max(xdata(:))]-xmean)*tan(atan2(-2*inv22(2),inv22(4)-inv22(1))/2)+ymean,'-','linewidth',2)

if (contains(textTH, 'chunk'))
%  mytext = text;
%  clear text;
  for yyyy = yStart:yEnd
    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
%  text = mytext;
%  disp(['hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',ydata(:)))]);
%  system(['echo hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',ydata(:))), ' >> corrtrend-hot']);
%  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',xdata(:)))]);
%  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',xdata(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(xdata(:),ydata(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
%title({titleArr{m}, ['r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
rng default;
title({titleArr{m}, sprintf('r=%+.3f e^2=%.3f',corr(xdata(:),ydata(:)),(crossval('mse',xdata,ydata,'Predfun',regf,'kfold',3,'mcreps',200))/e99)});  % ,'interpreter','none'
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);


end % textWgt loop

%system(['ps2pdf ',fn_figure]);  % TODO
%system(['rm ',fn_figure]);
%

%% automated plot polyfit sector
%
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
load(['temp_',verX,'_',caseid,'.mat'],'thresh','ds','yStart','yEnd','nyr','latt42','lont42','areaEarth','nd_jja','nd_djf' ,'lsm_jja');

fn_figure = ['panel_',verX,'_',caseid,'.ps'];
%system(['rm ',fn_figure]);  % TODO

%vernumArr = {'2a71','2a68','2a69','0a13','4a26','8a14','9a00',verX};  % 180627
%titleArr = {'DG83,prst=.9375,A=2.5','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA','Extreme'};

lat_xy = repmat(latt42',[ds(1) 1]);
lon_xy = repmat(lont42,[1 ds(2)]);
nr=1; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=346|lont42<=191,:,nr)=true; regArr{nr}='Eurasia';
nr=2; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=191&lont42<=346,:,nr)=true; regArr{nr}='N. America';
nr=3; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=30&lont42<=60,latt42>=45&latt42<=75,nr)=true; regArr{nr}='Russia';
nr=4; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=0&lont42<=30,latt42>=45&latt42<=75,nr)=true; regArr{nr}='Europe';

for nr=1:size(mask_xyr,3)

textWgtArr = {'chunk','lndchunk'};
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];
textTH = textTH{1};

% Hot
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
%ver = [vernumArr{end},'_',caseid];
if (contains(textTH, 'wgt'))
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','PERwjja_n');
  ydata = PERwjja_n;
else
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','PER_jja');
  ydata = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*mask_xyr(:,:,nr),[ds(1:2),nd_jja,nyr]),3),1))]';
end

  for m = 1:length(vernumArr)-1  %TODO

ver = [vernumArr{m},'_',caseid];
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
switch textTH
 case 'chunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PER_jja');
  xdata = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*mask_xyr(:,:,nr),[ds(1:2),nd_jja,nyr]),3),1))]';
  strT = 'JJA blocking area (km^2)';
 case 'lndchunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PER_jja');
  xdata = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*lsm_jja.*mask_xyr(:,:,nr),[ds(1:2),nd_jja,nyr]),3),1))]';
  strT = 'JJA blocking area on land (km^2)';
 case 'chunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERwjja_n');
  xdata = PERwjja_n;
  strT = 'Weighted JJA blocking area (unit*km^2)';
 case 'lndchunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERlwjja_n');
  xdata = PERlwjja_n;
  strT = 'Weighted JJA blocking area on land (unit*km^2)';
 otherwise
  warning('unexpected textth'); return;
end

subplot(3,4,m);  %TODO
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',xdata(:)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart:yEnd)',ydata(:)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
%  xlabel({'Weighted blocking area (unit*km^2)',strXCorr});
  ylabel({'Weighted extreme area (K*km^2)',strYCorr});
else
%  xlabel({'Blocking area (km^2)',strXCorr});
  ylabel({'Extreme area (km^2)',strYCorr});
end
xlabel({strT,strXCorr});
%plot(xdata(:),ydata(:),'.','markersize',3);
%plot([min(xdata(:)),max(xdata(:))],[min(xdata(:)),max(xdata(:))]*hotstat(1)+hotstat(2),'-','linewidth',2);
fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
% http://www.originlab.com/doc/Origin-Help/LR-Algorithm   confidence ellipse
%inv22 = inv( cov(xdata(:),ydata(:)) );  % normalize by N-1
%xmean=mean(xdata(:)); ymean=mean(ydata(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(xdata(:)),max(xdata(:)), min(ydata(:)),max(ydata(:))],'r','linewidth',2,'LevelList',1);
%plot([min(xdata(:)),max(xdata(:))], ([min(xdata(:)),max(xdata(:))]-xmean)*tan(atan2(-2*inv22(2),inv22(4)-inv22(1))/2)+ymean,'-','linewidth',2)

if (contains(textTH, 'chunk'))
%  mytext = text;
%  clear text;
  for yyyy = yStart:yEnd
    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
%  text = mytext;
%  disp(['hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',ydata(:)))]);
%  system(['echo hot trend ',textTH,'_',caseid,':  ',num2str(corr((yStart:yEnd)',ydata(:))), ' >> corrtrend-hot']);
%  disp(['blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',xdata(:)))]);
%  system(['echo blockJJA trend ',textTH,'_',ver,':  ',num2str(corr((yStart:yEnd)',xdata(:))), ' >> corrtrend-hot']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(xdata(:),ydata(:));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({titleArr{m}, ['hot, r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
if (m==1) title({regArr{nr}, titleArr{m}, ['hot, r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none'); end
axis square tight; %axis([min(xdata),max(xdata), min(ydata),max(ydata)]);
ylim([min(ydata),max(ydata)]); try xlim([min(xdata),max(xdata)]); end

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);


end % textWgt loop
end % nr nRegion loop

%system(['ps2pdf ',fn_figure]);  % TODO
%system(['rm ',fn_figure]);
%

%% polyfit after pointwise renormalize
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
%clearvars -except  verX caseid;
%miscArr = {'2a73','tuned DG83p',50.; '2a75','tuned DG83',75.; '0a15','tuned D13',75.; '0a13','D13',20; '4a29','tuned S06',75.; '4a26','S06',50; '8a15','tuned M13',50.; '8a14','M13',50; '9a07','tuned M17',75.; '9a06','M17',75; verX,'Extreme',50.};  % 181124
%vernumArr = miscArr(:,1);

load(['temp_',verX,'_',caseid,'.mat'],'thresh','ds','yStart','yEnd','nyr','latt42','lont42','areaEarth','nd_jja','nd_djf' ,'lsm_jja');
m=3; n=7;
%textTH='lndchunk';
ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PER_jja');
  PERjjam_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERstdm_jja = std(PERjjam_xyn,[],3);
ver = [vernumArr{n},'_',caseid];
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PER_jja');
  PERjjan_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  PERstdn_jja = std(PERjjan_xyn,[],3);

for j=ds(2)/2:ds(2)
 for i=1:ds(1)
  if (~lsm_jja(i,j)) continue; end
  corr_xy(i,j) = corr(squeeze(PERjjam_xyn(i,j,:)),squeeze(PERjjan_xyn(i,j,:)),'Type','Pearson');
%  wrk = nan(7*7,1);
%  for jo = -3:3
%  for io = -3:3
%    wrk( (jo+3)*7+io+4 ) = corr(squeeze(PERjja_xyn(1+mod(i+io-1,ds(1)),1+mod(j+jo-1,ds(2)),:)),squeeze(Hot_xyn(i,j,:)),'Type','Pearson');
%  end
%  end
%  if (all(isnan(wrk))) continue; end
%  [rmax_xym(i,j,m),pos] = max(wrk);
%  imax_xym(i,j,m) = -3 +mod(pos-1,7);
%  jmax_xym(i,j,m) = -3 +floor((pos-1)/7);
 end
% histc_ry(:,j) = histcounts(corr_xym(:,j,m),[-1:0.1:1]);
end
load coastlines  % for plotting
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%set(gca,'color',[0.5 0.5 0.5]);
%set(gcf,'InvertHardcopy','off');
patchm(-90+[-180:180]*0,[-180:180],[0.5 0.5 0.5]);
pcolormPC(latt42,lont42, corr_xy(:,:)'); %shading flat;
%colormap(gca,b2r(-1,1)); %colorbar;
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],20)); caxis([0 1]); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
%if (contains(textTH, 'wgt'))
%  title({titleArr{m},'JJA Pearson''s rho: weighted'},'fontsize',11);
%else
%  title({titleArr{m},'JJA Pearson''s rho: unweighted'},'fontsize',11);
%end
tightmap;
colorbar;

  m_n = [areaEarth * squeeze(mean(PERjjam_xyn.*lsm_jja,1))]';
  n_n = [areaEarth * squeeze(mean(PERjjan_xyn.*lsm_jja,1))]';
  corr(m_n,n_n)

  lsm_mn = lsm_jja&(PERstdm_jja>0)&(PERstdn_jja>0);
  [ nnz(lsm_jja), nnz(lsm_jja.*(PERstdm_jja==0)), nnz(lsm_jja.*(PERstdn_jja==0)), nnz(lsm_jja.*(lsm_mn==0)) ]
  m_n = [areaEarth * squeeze(mean(PERjjam_xyn.*lsm_mn,1))]';
  n_n = [areaEarth * squeeze(mean(PERjjan_xyn.*lsm_mn,1))]';
  corr(m_n,n_n)

  PERjjam_xyn(lsm_mn) = PERjjam_xyn(lsm_mn).*sqrt(PERstdn_jja(lsm_mn)./PERstdm_jja(lsm_mn));
  PERjjan_xyn(lsm_mn) = PERjjan_xyn(lsm_mn).*sqrt(PERstdm_jja(lsm_mn)./PERstdn_jja(lsm_mn));
  m_n = [areaEarth * squeeze(mean(PERjjam_xyn.*lsm_mn,1))]';
  n_n = [areaEarth * squeeze(mean(PERjjan_xyn.*lsm_mn,1))]';
  corr(m_n,n_n)


%% automated Block Freq & std (Pfahl2a in xtrm_colocate_pchan)
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
load(['temp_',verX,'_',caseid,'.mat'],'thresh','ds','yStart','yEnd','nyr','latt42','lont42','nd_jja','nd_djf');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panel_',verX,'_',caseid,'.ps'];
%system(['rm ',fn_figure]);

%vernumArr = {'5600','4002','8000','0602','2631','x600'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%vernumArr = {'2a53','2a64',verX};  % 180112
%vernumArr = {'2a66','2a70','2a67',verX};  % 180125
%titleArr = {'D=5,A=1.5','prst=.9375,A=2.5','D=1,A=0','Extreme'};
%vernumArr = {'8a10','8a11','8a12',verX};  % 180306
%titleArr = {'8a10','8a11','8a12','Extreme'};
%vernumArr = {'2a66','2a70','2a67','4a25','8a13',verX};  % 180306
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','S06,D=1,GHGN=-10','M13,D=1','Extreme'};
%lims = [35,35,100,50,50,35];
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
textWgtArr = {'chunk'};
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];

% JJA
%
figure('units','inches','position',[1 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
%disp('m1');
load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
%disp('m2'); % load 2nd slow

if (contains(textTH, 'wgt'))
  PERfreq_jja = mean(PER_jja.*(Wgt_jja),3);
else
  PERfreq_jja = mean(PER_jja,3);
end
%PERfreq_jja(PERfreq_jja==0) = nan;
if (isfield(prm,'yN1'))
  PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
else
  PERfreq_jja(:,[1:end/2]) = nan;
end

f_tp = @(a,b,c) ceil(c/a) +b*mod(c-1,a);
subplot(3,4,m);  %TODO
%subplot(3,4,f_tp(3,4,m));
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_jja'); shading flat;
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],10)); caxis([0 111]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
if (contains(textTH, 'wgt'))
  title({titleArr{m},'JJA weighted frequency (unit*%)'},'fontsize',11);
else
  title({titleArr{m},'JJA frequency (%)'},'fontsize',11);
end
tightmap;
%disp('m4');

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);
%

figure('units','inches','position',[1 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
%disp('m1');
load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
%disp('m2'); % load 2nd slow

if (contains(textTH, 'wgt'))
  PERstd_jja = std(mean(reshape(PER_jja.*(Wgt_jja),[ds(1:2),nd_jja,nyr]),3),[],4);
else
  PERstd_jja = std(mean(reshape(PER_jja,[ds(1:2),nd_jja,nyr]),3),[],4);
end
%PERstd_jja(PERstd_jja==0) = nan;
if (isfield(prm,'yN1'))
  PERstd_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
else
  PERstd_jja(:,[1:end/2]) = nan;
end

f_tp = @(a,b,c) ceil(c/a) +b*mod(c-1,a);
subplot(3,4,m);  %TODO
%subplot(3,4,f_tp(3,4,m));
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERstd_jja'); shading flat;
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],10)); caxis([0 111]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
if (contains(textTH, 'wgt'))
  title({titleArr{m},'JJA weighted frequency std (unit*%)'},'fontsize',11);
else
  title({titleArr{m},'JJA frequency std (%)'},'fontsize',11);
end
tightmap;
%disp('m4');

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);
%

end % textWgt loop

%system(['ps2pdf ',fn_figure]);  % TODO
%system(['rm ',fn_figure]);
%

%% automated time series (aka legend) & table of correlation 20180404
%
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
load(['temp_',verX,'_',caseid,'.mat'],'yStart','yEnd');

fn_figure = ['panel_',verX,'_',caseid,'.ps'];
%system(['rm ',fn_figure]);  % TODO

%vernumArr = {'5600','4002','8000','0602','2631'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%vernumArr = {'5900','0a02','2937','4923','8900','5940','2946','4943','x900'};  % 170623
%vernumArr = {'5900','0a02','2937','4923','8900','5940','2946','4943'};  % 170623
%vernumArr = {'2a66','2a70','2a67','0a12','4a26','4a25','8a13','9a00',verX};  % 180404
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','S06,D=1,GHGN=-10','M13,D=1','FALWA','Extreme'};
%vernumArr = {'2a66','2a70','2a67','0a12','4a26','4a25','8a13','9a00','x905','x906','x907','x908'};  % 180405 TODO
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','S06,D=1,GHGN=-10','M13,D=1','FALWA','Extreme1.5','Extreme2','Extreme2.5','Extreme3'};

%textWgtArr = {'chunk','lndchunk','chunkwgt','lndchunkwgt'};
textWgtArr = {'chunk','lndchunk'};
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];
textTH = textTH{1};

strLegend = cell(length(vernumArr),1);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
%  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle');

%bjjaArr(:,m) = bjjaArea_t(:);
%bdjfArr(:,m) = bdjfArea_t(:);
%if (contains(textTH, 'wgt'))
%  bjjaArr(:,m) = bjjaArr(:,m) / mean(bjjaArr(:,m));
%  bdjfArr(:,m) = bdjfArr(:,m) / mean(bdjfArr(:,m));
%end
switch textTH
 case 'chunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERjja_n');
  bjjaArr(:,m) = PERjja_n/mean(PERjja_n);
  bjjaArr0(:,m) = bjjaArr(:,m);
  strT = 'Mean JJA blocking area (normalized)';%(km^2)
 case 'lndchunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERljja_n');
  bjjaArr(:,m) = PERljja_n/mean(PERljja_n);
  bjjaArr1(:,m) = bjjaArr(:,m);
  strT = 'Mean JJA blocking area on land (normalized)';%(km^2)
 case 'chunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERwjja_n');
  bjjaArr(:,m) = PERwjja_n/mean(PERwjja_n);
  strT = 'Mean weighted JJA blocking area (normalized)';
 case 'lndchunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERlwjja_n');
  bjjaArr(:,m) = PERlwjja_n/mean(PERlwjja_n);
  strT = 'Mean weighted JJA blocking area on land (normalized)';
 otherwise
  warning('unexpected textth')
end
strLegend{m} = strTitle;
  end  % m
%strLegend = {'P12','S06','M13','D13','H14','Extreme'};
%strLegend = {'P12','D13','H14','S06','M13','Modified P12','Modified H14','Modified S06','Extreme'};
%strLegend = {'P12','D13','H14','S06','M13','Modified P12','Modified H14','Modified S06'};
strLegend = titleArr;  %TODO

%{
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
% Hot
%subplot(1,2,1);
hold on;

%if (contains(textTH, 'wgt'))
%  xlabel({'year'}); title({'Mean weighted JJA blocking area (normalized)'});
%else
%  xlabel({'year'}); title({'Mean JJA blocking area (normalized)'});%(km^2)
%end
xlabel({'year'}); title({strT});
%plot([yStart:yEnd]', bjjaArr);
plot([yStart:yEnd]', bjjaArr(:,1:end-1),'linewidth',1);
plot([yStart:yEnd]', bjjaArr(:,end:end),'k','linewidth',2);

ax=gca; ax.ColorOrderIndex = 1;
  for m = 1:length(vernumArr)
    tmpstat = [polyfit((yStart:yEnd)',bjjaArr(:,m),1)];
   if m<=length(vernumArr)-1  %TODO
    fplot(@(x) polyval(tmpstat,x), [yStart yEnd], '--');
   else
    fplot(@(x) polyval(tmpstat,x), [yStart yEnd], 'k--');
   end
%    title(['Legend, r=',num2str(corr((yStart:yEnd)',bjjaArea_t(:)'),'%+.3f')]);
  end  % m
xlim([yStart yEnd]); %axis square;
legend(strLegend,'Location','eastoutside');
set(gca,'fontsize',20);

print(gcf, '-dpsc2','-append',fn_figure);
%}

%% table of correlation
rhojja = corr(bjjaArr);
disp(rhojja)
rhojja(repmat([1:length(vernumArr)]',[1 length(vernumArr)])<=repmat([1:length(vernumArr)],[length(vernumArr) 1])) = nan;
%rhodjf = corr(bdjfArr);
%disp(rhodjf)
%rhodjf(repmat([1:length(vernumArr)]',[1 length(vernumArr)])<=repmat([1:length(vernumArr)],[length(vernumArr) 1])) = nan;

% automated table of crossval significance 20180920
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rng default;
e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),bjjaArr(:,end),'kfold',3,'mcreps',200);
e99 = (squeeze(mean(e99arr,1)));
for m = 1:length(vernumArr)-1  %TODO
  rng default;
  e01arr(:,m) = crossval(errf,bjjaArr(:,m),bjjaArr(:,end),'kfold',3,'mcreps',200);
end
e01 = (squeeze(mean(e01arr,1)))./e99;

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
% JJA
%subplot(1,2,2);
pcolorPH(1:length(vernumArr),1:length(vernumArr), rhojja);
hold on;
cm=colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));  %parula(100);
image(1:length(vernumArr)-1,length(vernumArr)+1.5,reshape(cm(round(interp1([0 0.5 1 99],[1 1 50 50],e01)),:),1,length(vernumArr)-1,3));
pcolor([1:length(vernumArr)]-0.5,length(vernumArr)+[1:2], nan(2,length(vernumArr)));
shading faceted; %set(get(gca,'children'),'edgecolor',[1 1 1]*0.7);
%plot([0.5 length(vernumArr)-0.5 length(vernumArr)-0.5 0.5 0.5],length(vernumArr)+[0.5 0.5 -0.5 -0.5 0.5],'k-','linewi',3);
%colormap(gca,b2r(-1,1)); %colorbar;
colormap(gca,colormap_CD([0.18 0.95],[1 0.35],[0],20)); caxis([0 1]); %colorbar;
%cvtcdf = @(earr) mean( 0.5*(1+sign(earr)) );
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(bjjaArr(:,end))/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
  for m = 1:length(vernumArr)
   if (m<length(vernumArr)) text(double(m),length(vernumArr)+1.5, sprintf('%.2f',e01(m)) ,'HorizontalAlignment','center','fontsize',16); end
%  for n = m+1:length(vernumArr)
  for n = 1:length(vernumArr)
   if (n>m) text(double(m),double(n), sprintf('%+.1f',rhojja(n,m)) ,'HorizontalAlignment','center','fontsize',16); end

   if (n<m & m<length(vernumArr))
%     rng default;
%     sigjja(n,m) = sum( bootci(2000,{@(x,y1,y0) corr(x,y1)-corr(x,y0),bjjaArr(:,end),bjjaArr(:,m),bjjaArr(:,n)})>0 )-1;
%     sigjja(n,m) = sum( 0.5*sign(bootci(2000,{@(x,y1,y0) corr(x,y1)-corr(x,y0),bjjaArr(:,end),bjjaArr(:,m),bjjaArr(:,n)})) );
     sigjja(n,m) = cvtcdf(-e01arr(:,n)+e01arr(:,m));
%     text(double(m),double(n), sprintf('%+i',sigjja(n,m)) ,'HorizontalAlignment','center','fontsize',16);
%     if (sigjja(n,m)==1) text(double(m),double(n), '\downarrow' ,'HorizontalAlignment','center','fontsize',24); end
%     if (sigjja(n,m)==-1) text(double(m),double(n), '\leftarrow' ,'HorizontalAlignment','center','fontsize',24); end
%     if (abs(norminv(sigjja(n,m)))>=1) image(double(m),double(n), reshape([min(1,2*sigjja(n,m));1-2*abs(sigjja(n,m)-0.5);min(1,2-2*sigjja(n,m))],1,1,3));  text(double(m),double(n), sprintf('%.2f',sigjja(n,m)),'HorizontalAlignment','center','fontsize',16,'color','w'); end
     if (abs(sigjja(n,m)-0.5)>0.4) image(double(m),double(n), reshape([min(1,2*sigjja(n,m));1-2*abs(sigjja(n,m)-0.5);min(1,2-2*sigjja(n,m))],1,1,3));  text(double(m),double(n), sprintf('%.2f',sigjja(n,m)),'HorizontalAlignment','center','fontsize',16,'color','w'); end
   end
  end
  end
title(['Correlation between indices: summer ',textTH],'fontsize',20);
xticks(1:length(vernumArr)); xticklabels(strLegend); xtickangle(45);
yticks([1:length(vernumArr),length(vernumArr)+1.5]); yticklabels([strLegend;'e^2 with extreme']);
axis equal; axis ij; axis([0.5 length(vernumArr)+0.5 0.5 length(vernumArr)+2]);
set(gca,'fontsize',18);

print(gcf, '-dpsc2','-append',fn_figure);


end % textWgt loop

clear bjjaArr
%for m = 1:length(vernumArr)-1  %TODO
%  sigjja2(1:2,m) = bootci(2000,{@corr,bjjaArr1(:,end),bjjaArr1(:,m)});
%  sigjja2(3,m) = sum( bootci(2000,{@(x,y1,y0) corr(x,y1)-corr(x,y0),bjjaArr1(:,end),bjjaArr1(:,m),bjjaArr0(:,m)})>0 )-1;
%end
%disp(sigjja2);
%
set(gcf,'units','inches','position',[0 1 10 9], 'paperUnits','inches','papersize',[10 9],'paperposition',[0 0 10 9]);
title('');
%set(gca,'FontSize',16);
fn_jpg = ['fig3_',verX,'.jpg'];
print(gcf, '-djpeg',fn_jpg);
%

%% automated table of crossval significance 20180920
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rng default;
e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),bjjaArr1(:,end),'kfold',3,'mcreps',200);
e99 = (squeeze(mean(e99arr,1)));
for m = 1:length(vernumArr)-1  %TODO
  rng default;
  e00arr(:,m) = crossval(errf,bjjaArr0(:,m),bjjaArr1(:,end),'kfold',3,'mcreps',200);
  rng default;
  e01arr(:,m) = crossval(errf,bjjaArr1(:,m),bjjaArr1(:,end),'kfold',3,'mcreps',200);
end
e00 = (squeeze(mean(e00arr,1)))./e99;
e01 = (squeeze(mean(e01arr,1)))./e99;

cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(bjjaArr1(:,end))/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
  for m = 1:length(vernumArr)-1
  for n = 1:length(vernumArr)-1
      sigjja(n,m) = cvtcdf(-e01arr(:,m)+e01arr(:,n));
  end
  end
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
%pcolorPH(1:length(vernumArr)-1,1:length(vernumArr)-1, (sigjja>normcdf(1)) - (sigjja<normcdf(-1)));
pcolorPH(1:length(vernumArr)-1,1:length(vernumArr)-1, sigjja);
shading faceted;
colormap(gca,b2r(-1,1)); caxis([0 1]); %colorbar;
  for m = 1:length(vernumArr)-1
  for n = 1:length(vernumArr)-1
   if (n>m)
     if (abs(norminv(sigjja(n,m)))>=1)
       text(double(m),double(n), sprintf('%.2f',sigjja(n,m)) ,'HorizontalAlignment','center','fontsize',16,'FontWeight','bold');
     else
       text(double(m),double(n), sprintf('%.2f',sigjja(n,m)) ,'HorizontalAlignment','center','fontsize',16);
     end
   end
   if (n<m)
     if (sigjja(n,m)>=normcdf(1)) text(double(m),double(n), '\downarrow' ,'HorizontalAlignment','center','fontsize',24); end
     if (sigjja(n,m)<=normcdf(-1)) text(double(m),double(n), '\leftarrow' ,'HorizontalAlignment','center','fontsize',24); end
   end
  end
  end
title(['Prob. of column index better than row index: summer ',textTH],'fontsize',20);
xticks(1:length(vernumArr)); xticklabels(strLegend); xtickangle(45);
yticks(1:length(vernumArr)); yticklabels(strLegend);
axis square; axis ij;
set(gca,'fontsize',18);

print(gcf, '-dpsc2','-append',fn_figure);

disp(sigjja);
e00, e01,
cvtcdf(-e01arr+repmat(e99arr,[1 length(vernumArr)-1]))
cvtcdf(-e01arr+e00arr)

%% regress on two predictors
rhojja = corr(bjjaArr1);
  for m = 1:length(vernumArr)-1
  for n = m+1:length(vernumArr)-1
    inv(rhojja([m,n,end],[m,n,end]));
    rhojja2(n,m) = sqrt(1-1/ans(end));
  end
  end



%system(['ps2pdf ',fn_figure]);  % TODO
%system(['rm ',fn_figure]);
%

%% automated pointwise corr 20180606
%
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
load(['temp_',verX,'_',caseid,'.mat'],'thresh','ds','yStart','yEnd','nyr','latt42','lont42','areaEarth','nd_jja','nd_djf' ,'lsm_jja');
%nyr=nyr*nd_jja; nd_jja=1;

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panel_',verX,'_',caseid,'.ps'];
%system(['rm ',fn_figure]);  % TODO

%vernumArr = {'5600','4002','8000','0602','2631'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633'};
%vernumArr = {'5900','4923','8900','0902','2933'};
%vernumArr = {'5900','4923','8900','0a02','2937'};  % 170621
%vernumArr = {'5940','4943','2946','x900'};  % cutoff low
%vernumArr = {'2a66','2a70','2a67','4a25','8a13',verX};  % 180306
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','S06,D=1,GHGN=-10','M13,D=1','Extreme'};
%vernumArr = {'2a66','2a70','2a67','0a12','4a26','4a25','8a13','9a00',verX};  % 180404
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','S06,D=1,GHGN=-10','M13,D=1','FALWA','Extreme'};
%vernumArr = {'2a66','2a70','2a67','0a12','4a26','4a25','8a13','9a00','x905','x906','x907','x908'};  % 180405
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','S06,D=1,GHGN=-10','M13,D=1','FALWA','Extreme1.5','Extreme2','Extreme2.5','Extreme3'};
%vernumArr = {'2a71','2a68','2a69','0a13','4a26','8a14','9a00',verX};  % 180627
%titleArr = {'DG83,prst=.9375,A=2.5','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA','Extreme'};
%vernumArr = {'2a73','2a68','2a69','0a13','4a26','8a14','9a06',verX};  % 180809
%titleArr = {'DG83,prst=.5,A=1','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA,D=5,A=median','Extreme'};
%vernumArr = vernumArr(end-1:end);
%titleArr = titleArr(end-1:end);%([1,end]);%

%textWgtArr = {'chunk','chunkwgt'};
textWgtArr = {'chunk'};
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];
textTH = textTH{1};

% Hot
%fzero(@(r) 2*(1-tcdf( r*sqrt((nyr-2)/(1-r^2)), nyr-2 ))-0.05, 0.5)
rsig = fzero(@(r) r*sqrt((nyr-2)/(1-r^2)) -tinv(0.975,nyr-2), 0.5);
area_jja = lsm_jja*diag(areaEarth)/ds(1);
area_ll = area_jja(lsm_jja)*area_jja(lsm_jja)';
%sum(HotCov_ll(:).*area_ll(:))
lat_xy = repmat(latt42',[ds(1) 1]);
lon_xy = repmat(lont42,[1 ds(2)]);
dis_ll = distance(repmat(lat_xy(lsm_jja),1,nnz(lsm_jja)),repmat(lon_xy(lsm_jja),1,nnz(lsm_jja)),repmat(lat_xy(lsm_jja),1,nnz(lsm_jja))',repmat(lon_xy(lsm_jja),1,nnz(lsm_jja))');
i_dis = discretize(dis_ll(:),[0:2:100]);

area2_dis = nan(50,1);
for ct=1:50
  area2_dis(ct) = sum(area_ll(i_dis==ct));
end

%figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
%ver = [vernumArr{end},'_',caseid];
load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf' ,'PERjja_n');
if (contains(textTH, 'wgt'))
  Hot_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  Hot_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  ydata = PERjja_n;
end
%Hot_xyn = ( Hot_xyn+circshift(Hot_xyn,-1,1)+circshift(Hot_xyn,1,1)+circshift(Hot_xyn,-2,1)+circshift(Hot_xyn,2,1) )/5;
%Hot_xyn = ( Hot_xyn+circshift(Hot_xyn,-1,1)+circshift(Hot_xyn,1,1) )/3;
%Hot_xyn = ( Hot_xyn+circshift(Hot_xyn,-1,1)+circshift(Hot_xyn,1,1) )/3;
%Hot_xyn = ( Hot_xyn+circshift(Hot_xyn,-1,1)+circshift(Hot_xyn,1,1) )/3;
%Hot_xyn = ( Hot_xyn+circshift(Hot_xyn,-1,1)+circshift(Hot_xyn,1,1) )/3;
%Hot_xyn(:,2:end-1,:) = ( Hot_xyn(:,1:end-2,:)+Hot_xyn(:,2:end-1,:)+Hot_xyn(:,3:end,:) )/3;
%Hot_xyn(:,2:end-1,:) = ( Hot_xyn(:,1:end-2,:)+Hot_xyn(:,2:end-1,:)+Hot_xyn(:,3:end,:) )/3;
%Hot_xyn(:,2:end-1,:) = ( Hot_xyn(:,1:end-2,:)+Hot_xyn(:,2:end-1,:)+Hot_xyn(:,3:end,:) )/3;
%Hot_xyn(:,2:end-1,:) = ( Hot_xyn(:,1:end-2,:)+Hot_xyn(:,2:end-1,:)+Hot_xyn(:,3:end,:) )/3;
Hot_ln = reshape(Hot_xyn,ds(1)*ds(2),nyr); Hot_ln = Hot_ln(lat_xy>thresh{5},:); %Hot_ln = Hot_ln(lsm_jja,:);
Hot_ln = Hot_ln-repmat(mean(Hot_ln,2),[1 nyr]);
Hot_ln = Hot_ln./repmat(sqrt(sum(Hot_ln.^2,2)),[1 nyr]);
%Hot_ln = Hot_ln./norm(ydata-mean(ydata));

cov_dm = nan(50,length(vernumArr));
ppcov_dm = nan(50,length(vernumArr));
corr_xym = nan([ds(1:2),length(vernumArr)]);
rmax_xym = nan([ds(1:2),length(vernumArr)]);
imax_xym = nan([ds(1:2),length(vernumArr)]);
jmax_xym = nan([ds(1:2),length(vernumArr)]);
  for m = 1:length(vernumArr)-0  %TODO

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf' ,'PERljja_n');
if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
  xdata = PERljja_n;
  ydata = PERjja_n;  % ydata reused...
  rCorr_m(m) = corr(xdata(:),ydata(:));
  PERstd_jja = std(mean(reshape(PER_jja,[ds(1:2),nd_jja,nyr]),3),[],4);
end
PERjja_xyn(:,1:end/2,:)=0;

%PERjja_xyn = ( PERjja_xyn+circshift(PERjja_xyn,-1,1)+circshift(PERjja_xyn,1,1)+circshift(PERjja_xyn,-2,1)+circshift(PERjja_xyn,2,1) )/5;
%PERjja_xyn(:,2:end-1,:) = ( PERjja_xyn(:,1:end-2,:)+PERjja_xyn(:,2:end-1,:)+PERjja_xyn(:,3:end,:) )/3;

PERjja_ln = reshape(PERjja_xyn,ds(1)*ds(2),nyr); PERjja_ln = PERjja_ln(lat_xy>thresh{5},:); %PERjja_ln = PERjja_ln(lsm_jja,:);
PERjja_ln = PERjja_ln-repmat(mean(PERjja_ln,2),[1 nyr]);
PERjja_ln = PERjja_ln./repmat(sqrt(sum(PERjja_ln.^2,2)),[1 nyr]);
%PERjja_ln = PERjja_ln./norm(PERljja_n-mean(PERljja_n));
HotCov_ll = Hot_ln * PERjja_ln';
ppCov_ll = PERjja_ln * PERjja_ln';

%for ct=1:50
%  cov_dm(ct,m) = sum(area_ll(i_dis==ct).*HotCov_ll(i_dis==ct));
%  ppcov_dm(ct,m) = sum(area_ll(i_dis==ct).*ppCov_ll(i_dis==ct));
%end

%disp('return'); return;

histc_ry = zeros(20,ds(2));
for j=ds(2)/2:ds(2)
 for i=1:ds(1)
  if (~lsm_jja(i,j)) continue; end
  corr_xym(i,j,m) = corr(squeeze(PERjja_xyn(i,j,:)),squeeze(Hot_xyn(i,j,:)),'Type','Pearson');
  wrk = nan(7*7,1);
%  for jo = -3:3
%  for io = -3:3
%    wrk( (jo+3)*7+io+4 ) = corr(squeeze(PERjja_xyn(1+mod(i+io-1,ds(1)),1+mod(j+jo-1,ds(2)),:)),squeeze(Hot_xyn(i,j,:)),'Type','Pearson');
%  end
%  end
  if (all(isnan(wrk))) continue; end
  [rmax_xym(i,j,m),pos] = max(wrk);
  imax_xym(i,j,m) = -3 +mod(pos-1,7);
  jmax_xym(i,j,m) = -3 +floor((pos-1)/7);
 end
 histc_ry(:,j) = histcounts(corr_xym(:,j,m),[-1:0.1:1]);
end
aCorr_rm(:,m) = histc_ry*areaEarth(:)/ds(1);

% corr estimation
%disp('return'); return;
area_jja = lsm_jja*diag(areaEarth)/ds(1);
%rCorr1_m(m) = nansum(corr_xy(:).*area_jja(:))./sum(area_jja(:));
rCorr1_m(m) = nansum(nansum(corr_xym(:,:,m).*area_jja,2),1)./sum(area_jja(:));
%rCorr1_m = squeeze(nansum(nansum(corr_xym.*area_jja,2),1))./sum(area_jja(:));
area_jja = lsm_jja*diag(areaEarth)/ds(1).*PERstd_jja;
%rCorr2_m(m) = nansum(corr_xy(:).*area_jja(:))./sum(area_jja(:));
rCorr2_m(m) = nansum(nansum(corr_xym(:,:,m).*area_jja,2),1)./sum(area_jja(:));

% hemispheric maximum
%{
corr_xy = nan(ds(1:2));
corr_xy(lat_xy>thresh{5}) = diag(HotCov_ll);
%histc_ry = zeros(20,ds(2));
for j=ds(2)/2:ds(2)
 for i=1:ds(1)
  if (~lsm_jja(i,j)) continue; end
%  corr_xym(i,j,m) = corr(squeeze(PERjja_xyn(i,j,:)),squeeze(Hot_xyn(i,j,:)),'Type','Pearson');
%  if (all(isnan(wrk))) continue; end
  [rmax_xym(i,j,m),pos] = max(HotCov_ll(i+ds(1)*(j-find(latt42>thresh{5},1,'first')),:));
  imax_xym(i,j,m) = 1 +mod(pos-1,ds(1));
  jmax_xym(i,j,m) = find(latt42>thresh{5},1,'first') +floor((pos-1)/ds(1));
 end
% histc_ry(:,j) = histcounts(corr_xym(:,j,m),[-1:0.1:1]);
end
%aCorr_rm(:,m) = histc_ry*areaEarth(:)/ds(1);
imax_xym(:,:,m) = -ds(1)/2+ mod(imax_xym(:,:,m)-repmat([1:ds(1)]',[1 ds(2)]) +ds(1)/2, ds(1));
jmax_xym(:,:,m) = jmax_xym(:,:,m) - repmat([1:ds(2)],[ds(1) 1]);
%}

% understand stripes
%{
if (m==1)
  iwrk_xym = 1+ mod(imax_xym+repmat([1:ds(1)]',[1 ds(2)])-1,ds(1));
  jwrk_xym = jmax_xym+repmat([1:ds(2)],[ds(1) 1]);
  wrk_xym = jwrk_xym*1000+iwrk_xym;
  wrk_xym = categorical(wrk_xym);
  catloc = categories(wrk_xym);
%  [aa,bb]=max(squeeze(sum(countcats(wrk_xym(:,:,1:end-1),1),2)));
  [aa,bb]=max(squeeze(sum(countcats(wrk_xym(:,:,m),1),2)));
  disp(aa), disp(catloc(bb));
%disp('return'); return;

%  close;
  figure(97);
  set(gcf,'units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
%  i=40; j=49;
%  i=41; j=53;  %m=2;  x9122a68
  i=mod(str2num(catloc{bb}),1000); j=floor((str2num(catloc{bb})-1)/1000);
%  disp(wrk_xym(i+[-3:3],j+[-3:3],m));
  subplot(2,1,1);
%  plot(squeeze(PERjja_xyn(i+[-3:3],j,:))');
%  legend;
  pcolorPH(yStart:yEnd,[-3:3],squeeze(PERjja_xyn(i+[-3:3],j,:))*nd_jja); colorbar;
  xlabel('year'); ylabel(sprintf('x relative to %.0fN %.0fE',latt42(j),lont42(i))); title(titleArr{m});
  set(gca,'fontsize',20);
  subplot(2,1,2);
%  plot(squeeze(Hot_xyn(i+[-3:3],j,:))');
%  legend;
  pcolorPH(yStart:yEnd,[-3:3],squeeze(Hot_xyn(i+[-3:3],j,:))*nd_jja); colorbar;
  xlabel('year'); ylabel(sprintf('x relative to %.0fN %.0fE',latt42(j),lont42(i))); title(titleArr{end});
  set(gca,'fontsize',20);

  disp([imax_xym(i+[-3:3],j,m) jmax_xym(i+[-3:3],j,m)]);
%  disp('return'); return;
  print(gcf, '-dpsc2','-append',fn_figure);

 % scatter
  ii = find(wrk_xym(i+[-3:-1],j,m)==catloc{bb},1) +i-4; ydata=Hot_xyn(ii,j,:);
  y2 = ydata(:)'-mean(ydata); y2=y2/norm(y2);
  x2 = squeeze(PERjja_xyn(i+[-3:3],j,:));
  figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  subplot(2,1,1);
  pcolorPH(yStart:yEnd,[-3:3],x2-mean(x2,2)-sum(x2.*y2,2)*y2);
  b2rCD(125); caxis([-.3 .3]); colorbar;
  xlabel('year'); ylabel(sprintf('x relative to %.0fN %.0fE',latt42(j),lont42(i))); title(['Residual, when regressed on extreme @x=',num2str(ii-i),' ',titleArr{m}]);
  set(gca,'fontsize',20);
%  subplot(2,1,2);
%  pcolorPH(yStart:yEnd,[-3:3],squeeze(Hot_xyn(i+[-3:3],j,:))*nd_jja); colorbar;
%  xlabel('year'); ylabel(sprintf('x relative to %.0fN %.0fE',latt42(j),lont42(i))); title(titleArr{end});
%  set(gca,'fontsize',20);
  print(gcf, '-dpsc2','-append',fn_figure);
 % scatter
  figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for ip=1:7
    subplot(2,4,ip); hold on;
    xdata=PERjja_xyn(i+ip-4,j,:);
    xlabel(titleArr{m}); ylabel([titleArr{end},' @x=',num2str(ii-i)]);
    fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
    fplot(@(y) polyval(polyfit(ydata,xdata,1),y), @(y)y, [min(ydata),max(ydata)],'-','linewidth',2);
    for yyyy = yStart:yEnd
      text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
    end;
    title({[sprintf('block x=%i, r=%+.3f',ip-4,corr(xdata(:),ydata(:)))]}, 'interpreter','none');
    axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);
  end %loop ip
  print(gcf, '-dpsc2','-append',fn_figure);
end
%}

  end  % m

% corr map
%
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for m = 1:length(vernumArr)-0  %TODO
f_tp = @(a,b,c) ceil(c/a) +b*mod(c-1,a);
subplot(3,4,m);  %TODO
%subplot(3,4,f_tp(3,4,m));
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
%set(gca,'color',[0.5 0.5 0.5]);
%set(gcf,'InvertHardcopy','off');
patchm(-90+[-180:180]*0,[-180:180],[0.5 0.5 0.5]);
pcolormPC(latt42,lont42, corr_xym(:,:,m)'); %shading flat;
%colormap(gca,b2r(-1,1)); %colorbar;
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],20)); caxis([0 1]); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
if (contains(textTH, 'wgt'))
  title({titleArr{m},'JJA Pearson''s rho: weighted'},'fontsize',11);
%title({titleArr{m}, ['hot, r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
else
  title({titleArr{m},'JJA Pearson''s rho: unweighted'},'fontsize',11);
end
tightmap;
  end  % m
colorbar;
print(gcf, '-dpsc2','-append',fn_figure);

% fig 98 pdf
%
figure(98);
set(gcf,'units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
if (contains(textTH, 'wgt'))
  subplot(1,2,2);
else
  subplot(1,2,1);
end
%cm=jetCD(ceil((length(vernumArr)-1)/2)); %jet(length(vernumArr));
%cm=colormap_CD([0.45 0.70; 0.25 0.9],[.35 .35],[0 0],ceil((length(vernumArr)-1)/2));
cm=colormap_CD([0.25 0.90; 0.35 0.7],[.35 .35],[0 0],ceil((length(vernumArr)-1)/2));
hold on;
ax=gca; ax.ColorOrder=cm;
ax.LineStyleOrder = {'-','--'}; % {'-','--','-.',':'};
  for m = 1:length(vernumArr)-1  %TODO
ax.LineStyleOrderIndex = m;
plot(-0.95:0.1:0.95,aCorr_rm(:,m));
  end  % m
xlabel('Pearson''s rho'); ylabel('Area (km^2)');
if (contains(textTH, 'wgt'))
  title('JJA PDF of correlation: weighted','fontsize',11);
else
  title('JJA PDF of correlation: unweighted','fontsize',11);
end
xlim([-1 1]);
set(gca,'FontSize',20);
legend(titleArr(1:end-1),'FontSize',12,'location','northwest');
%

% corr vs. est corr
%
subplot(2,2,2);
ydata=rCorr_m; ylabel('Hemispheric correlation');
xdata=rCorr1_m; xlabel('Mean local correlation weighted by area');
  for m = 1:length(vernumArr)-1  %TODO
    text(double(xdata(m)),double(ydata(m)), vernumArr{m},'HorizontalAlignment','center' ,'Color',cm(m,:));
  end
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);

subplot(2,2,4);
ydata=rCorr_m; ylabel('Hemispheric correlation');
xdata=rCorr2_m; xlabel('Mean local correlation weighted by area*std(block)');
  for m = 1:length(vernumArr)-1  %TODO
    text(double(xdata(m)),double(ydata(m)), vernumArr{m},'HorizontalAlignment','center' ,'Color',cm(m,:));
  end
axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);
print(gcf, '-dpsc2','-append',fn_figure);
%

% plot max corr offset
%{
imax_xym(rmax_xym<rsig)=nan;
jmax_xym(rmax_xym<rsig)=nan;
rmax_xym(rmax_xym<rsig)=nan;

figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for m = 1:length(vernumArr)-0  %TODO
subplot(3,4,m);
%subplot(3,4,f_tp(3,4,m));
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
%set(gca,'color',[0.5 0.5 0.5]);
%set(gcf,'InvertHardcopy','off');
patchm(-90+[-180:180]*0,[-180:180],[0.5 0.5 0.5]);
pcolormPC(latt42,lont42, rmax_xym(:,:,m)'); shading flat;
%colormap(gca,b2r(-1,1)); %colorbar;
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],20)); caxis([0 1]); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
if (contains(textTH, 'wgt'))
  title({titleArr{m},'JJA max Pearson''s rho: weighted'},'fontsize',11);
%title({titleArr{m}, ['hot, r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
else
  title({titleArr{m},'JJA max Pearson''s rho: unweighted'},'fontsize',11);
end
tightmap;
  end  % m
colorbar;
print(gcf, '-dpsc2','-append',fn_figure);

figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for m = 1:length(vernumArr)-0  %TODO
subplot(3,4,m);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
%set(gca,'color',[0.5 0.5 0.5]);
%set(gcf,'InvertHardcopy','off');
patchm(-90+[-180:180]*0,[-180:180],[0.5 0.5 0.5]);
pcolormPC(latt42,lont42, imax_xym(:,:,m)'); shading flat;
colormap(gca,b2r(-3,3)); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
  title({titleArr{m},'i-offset'},'fontsize',11);
tightmap;
  end  % m
colorbar;
print(gcf, '-dpsc2','-append',fn_figure);

figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
  for m = 1:length(vernumArr)-0  %TODO
subplot(3,4,m);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
%set(gca,'color',[0.5 0.5 0.5]);
%set(gcf,'InvertHardcopy','off');
patchm(-90+[-180:180]*0,[-180:180],[0.5 0.5 0.5]);
pcolormPC(latt42,lont42, jmax_xym(:,:,m)'); shading flat;
colormap(gca,b2r(-3,3)); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
  title({titleArr{m},'j-offset'},'fontsize',11);
tightmap;
  end  % m
colorbar;
print(gcf, '-dpsc2','-append',fn_figure);
%}

end % textWgt loop
%fig 98 print(gcf, '-dpsc2','-append',fn_figure);

%disp('return'); return;
%% save 2 nc for ncl 20181022
%
%vernumArr = {'2a72','2a73','0a13','0a15'};  % 181010
%textWgtArr = {'chunk','lndchunk','lndchunk','lndchunk'};
%titleArr = {'DG83 prst=0.94 A=3: land and ocean','DG83 prst=0.94 A=3: land only','D13'
%titleArr = {'DG83p: land and ocean','DG83p: land only','D13 on land: original threshold','D13 on land: tuned threshold'};

%PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
%PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
 fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
fn_savenc = ['pointcorr_',verX,'_',caseid,'.nc'];
 system(['rm ',fn_savenc]);
system(['ncks -v lat,lat_bnds,lon,lon_bnds ',fn_t42,' ',fn_savenc]);
nccreate(fn_savenc,'corr_xym','Dimensions',{'lon',ds(1),'lat',ds(2),'index',length(vernumArr)},'DataType','single')  %,'Format','netcdf4'
nccreate(fn_savenc,'titleArr','Dimensions',{'charlen',size(char(titleArr)',1),'index',length(vernumArr)},'DataType','char')
nccreate(fn_savenc,'rCorr_m','Dimensions',{'index',length(vernumArr)},'DataType','single')
nccreate(fn_savenc,'rCorr1_m','Dimensions',{'index',length(vernumArr)},'DataType','single')
nccreate(fn_savenc,'rCorr2_m','Dimensions',{'index',length(vernumArr)},'DataType','single')
ncwrite(fn_savenc,'corr_xym',corr_xym)
ncwrite(fn_savenc,'titleArr',char(titleArr)')
ncwrite(fn_savenc,'rCorr_m',rCorr_m)
ncwrite(fn_savenc,'rCorr1_m',rCorr1_m)
ncwrite(fn_savenc,'rCorr2_m',rCorr2_m)
system(['ln -sf ',fn_savenc,' pointcorr.nc']);
system(['ncl -Q xy.mp.grl_pointcorr.ncl']);

%% cov vs. distance
%{
close
cov_dm = cov_dm./repmat(area2_dis,[1 length(vernumArr)])*sum(area2_dis);
ppcov_dm = ppcov_dm./repmat(area2_dis,[1 length(vernumArr)])*sum(area2_dis);

figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
subplot(2,1,1);
hold on;
yyaxis left;
ax=gca; ax.ColorOrder=jet(length(vernumArr));
ax.LineStyleOrder = {'-','--'}; % {'-','--','-.',':'};
  for m = 1:length(vernumArr)  %TODO
ax.LineStyleOrderIndex = m;
plot(1:2:99,cov_dm(:,m));
  end  % m
ylim([-3 13]); xlim([0 100]);
xlabel('Big circle distance (deg)'); ylabel('Covariance with extreme (scaled)');
grid on;
yyaxis right;
plot(1:2:99,area2_dis,'DisplayName',''); ylabel('Area^2 in bin'); ylim([0 max(area2_dis)]);
legend(titleArr(1:end),'FontSize',12,'location','northeast');
%if (contains(textTH, 'wgt'))
%  title('JJA PDF of correlation: weighted','fontsize',11);
%else
%  title('JJA PDF of correlation: unweighted','fontsize',11);
%end
set(gca,'FontSize',20);

subplot(2,1,2);
hold on;
ax=gca; ax.ColorOrder=jet(length(vernumArr));
ax.LineStyleOrder = {'-','--'}; % {'-','--','-.',':'};
  for m = 1:length(vernumArr)  %TODO
ax.LineStyleOrderIndex = m;
plot(1:2:99,cov_dm(:,m));
  end  % m
ylim([-3 2]); xlim([0 100]);
xlabel('Big circle distance (deg)'); ylabel('Covariance with extreme (scaled)');
grid on;
%legend(titleArr(1:end),'FontSize',12,'location','northeast');
set(gca,'FontSize',20);
print(gcf, '-dpsc2','-append',fn_figure);

figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
subplot(2,1,1);
hold on;
yyaxis left;
ax=gca; ax.ColorOrder=jet(length(vernumArr));
ax.LineStyleOrder = {'-','--'}; % {'-','--','-.',':'};
  for m = 1:length(vernumArr)  %TODO
ax.LineStyleOrderIndex = m;
plot(1:2:99,ppcov_dm(:,m));
  end  % m
%axis tight; xlim([0 100]);
xlabel('Big circle distance (deg)'); ylabel('Self covariance (scaled)');
grid on;
yyaxis right;
plot(1:2:99,area2_dis,'DisplayName',''); ylabel('Area^2 in bin'); ylim([0 max(area2_dis)]);
legend(titleArr(1:end),'FontSize',12,'location','northeast');
set(gca,'FontSize',20);

subplot(2,1,2);
hold on;
ax=gca; ax.ColorOrder=jet(length(vernumArr));
ax.LineStyleOrder = {'-','--'}; % {'-','--','-.',':'};
  for m = 1:length(vernumArr)  %TODO
ax.LineStyleOrderIndex = m;
plot(1:2:99,ppcov_dm(:,m));
  end  % m
ylim([-6 4]);
xlabel('Big circle distance (deg)'); ylabel('Self covariance (scaled)');
grid on;
set(gca,'FontSize',20);
print(gcf, '-dpsc2','-append',fn_figure);
%}




%% Spearman
%{
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];
textTH = textTH{1};

% Hot
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
%ver = [vernumArr{end},'_',caseid];
load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
if (contains(textTH, 'wgt'))
  Hot_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  Hot_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
end

  for m = 1:length(vernumArr)-1  %TODO

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
end

histc_ry = zeros(20,ds(2));
for j=ds(2)/2:ds(2)
 for i=1:ds(1)
  corr_xym(i,j,m) = corr(squeeze(PERjja_xyn(i,j,:)),squeeze(Hot_xyn(i,j,:)),'Type','Spearman');
 end
 histc_ry(:,j) = histcounts(corr_xym(:,j,m),[-1:0.1:1]);
end
aCorr_rm(:,m) = histc_ry*areaEarth(:)/ds(1);
%PERfreq_jja(PERfreq_jja==0) = nan;
%if (isfield(prm,'yN1'))
%  PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
%else
%  PERfreq_jja(:,[1:end/2]) = nan;
%end

f_tp = @(a,b,c) ceil(c/a) +b*mod(c-1,a);
subplot(3,4,m);  %TODO
%subplot(3,4,f_tp(3,4,m));
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, corr_xy'); shading flat;
colormap(gca,b2r(-1,1)); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
if (contains(textTH, 'wgt'))
  title({titleArr{m},'JJA Spearman''s rho: weighted'},'fontsize',11);
%title({titleArr{m}, ['hot, r=',num2str(corr(xdata(:),ydata(:)),'%+.3f')]}, 'interpreter','none');
else
  title({titleArr{m},'JJA Spearman''s rho: unweighted'},'fontsize',11);
end
tightmap;

  end  % m
print(gcf, '-dpsc2','-append',fn_figure);

figure(99);
set(gcf,'units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
if (contains(textTH, 'wgt'))
  subplot(1,2,2);
else
  subplot(1,2,1);
end
hold on;
ax=gca; ax.ColorOrder=jet(length(vernumArr)-1);
ax.LineStyleOrder = {'-','--'}; % {'-','--','-.',':'};
  for m = 1:length(vernumArr)-1  %TODO
ax.LineStyleOrderIndex = m;
plot(-0.95:0.1:0.95,aCorr_rm(:,m));
  end  % m
xlabel('Spearman''s rho'); ylabel('Area (km^2)');
if (contains(textTH, 'wgt'))
  title('JJA PDF of correlation: weighted','fontsize',11);
else
  title('JJA PDF of correlation: unweighted','fontsize',11);
end
xlim([-0.5 1]);
set(gca,'FontSize',20);
legend(titleArr(1:end-1),'FontSize',12,'location','northwest');

end % textWgt loop
print(gcf, '-dpsc2','-append',fn_figure);
%}


system(['ps2pdf ',fn_figure]);  % TODO
system(['rm ',fn_figure]);
%

%% automated SVD spatial: xtrm_scatter
%{
caseid=['ERA-interim_19790101-20171231'];
verX='x912';
load(['temp_',verX,'_',caseid,'.mat'],'thresh','ds','yStart','yEnd','nyr','latt42','lont42','areaEarth','nd_jja','nd_djf');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panel_',verX,'_',caseid,'.ps'];
%system(['rm ',fn_figure]);

%vernumArr = {'5600','4002','8000','0602','2631','x600'};  % qual
%vernumArr = {'5600','4023','8000','0602','2633','x600'};
%vernumArr = {'5900','4923','8900','0902','2933','x900'};
%vernumArr = {'5900','4923','8900','0a02','2937','x900'};  % 170621
%vernumArr = {'2a53','2a64',verX};  % 180112
%vernumArr = {'2a66','2a70','2a67',verX};  % 180125
%titleArr = {'D=5,A=1.5','prst=.9375,A=2.5','D=1,A=0','Extreme'};
%vernumArr = {'8a10','8a11','8a12',verX};  % 180306
%titleArr = {'8a10','8a11','8a12','Extreme'};
%vernumArr = {'2a66','2a70','2a67','4a25','8a13',verX};  % 180306
%titleArr = {'DG83,D=5,A=1.5','DG83,prst=.9375,A=2.5','DG83,D=1,A=0','S06,D=1,GHGN=-10','M13,D=1','Extreme'};
%lims = [35,35,100,50,50,35];
vernumArr = {'2a71','2a68','2a69','0a13','4a26','8a14','9a00',verX};  % 180627
titleArr = {'DG83,prst=.9375,A=2.5','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA','Extreme'};
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
%textWgtArr = {'chunk','chunkwgt'};
textWgtArr = {'chunk'};
for textTH = textWgtArr

%thresh{5} = 0;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];

% JJA
%
%figure('units','inches','position',[1 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf' ,'PERjja_n');
if (contains(textTH, 'wgt'))
  Hot_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
else
  Hot_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
end

  for m = 1:length(vernumArr)-1  %TODO

ver = [vernumArr{m},'_',caseid];
disp('m1');
load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
disp('m2'); % load 2nd slow

if (contains(textTH, 'wgt'))
  PERjja_xyn = squeeze(mean( reshape(PER_jja.*(Wgt_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  PERdjf_xyn = squeeze(mean( reshape(PER_djf.*(Wgt_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%  Hot_xyn = squeeze(mean( reshape(Hot_jja.*(mx2t_jja), ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  Cold_xyn = squeeze(mean( reshape(Cold_djf.*(mn2t_djf), ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
else
  PERjja_xyn = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  PERdjf_xyn = squeeze(mean( reshape(PER_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%  Hot_xyn = squeeze(mean( reshape(Hot_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%  Cold_xyn = squeeze(mean( reshape(Cold_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
end
PERjja_sn  =  reshape( PERjja_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
%PERdjf_sn  =  reshape( PERdjf_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);
Hot_sn  =  reshape( Hot_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr]), [],nyr) /ds(1);
%Cold_sn  =  reshape( Cold_xyn.*repmat( reshape(areaEarth,[1 ds(2) 1]), [ds(1) 1 nyr-1]), [],nyr-1) /ds(1);

% -mean
PERjja_sn = PERjja_sn -repmat(mean(PERjja_sn,2),[1 nyr]);  % double not needed..
%PERdjf_sn = PERdjf_sn -repmat(mean(PERdjf_sn,2),[1 nyr-1]);  % double not needed..
Hot_sn = Hot_sn -repmat(mean(Hot_sn,2),[1 nyr]);  % double not needed..
%Cold_sn = Cold_sn -repmat(mean(Cold_sn,2),[1 nyr-1]);  % double not needed..

%PER_yht(:,:,timeNan) = 0;

HotCov = Hot_sn * PERjja_sn';
%ColdCov = Cold_sn * PERdjf_sn';

[HotU, HotS, HotV] = svds(double(HotCov),4);  % svd-subset. U:extreme; V:PER
%[ColdU, ColdS, ColdV] = svds(double(ColdCov),4);  % svd-subset. U:extreme; V:PER

HotV = HotV * diag(sign(mean(HotU,1)));  HotU = HotU * diag(sign(mean(HotU,1)));
%ColdV = ColdV * diag(sign(mean(ColdU,1)));  ColdU = ColdU * diag(sign(mean(ColdU,1)));
HotPrct1 = diag(HotS).^2/norm(HotCov,'fro')^2 *100;
HotPrct2 = vecnorm(Hot_sn'*HotU).^2/norm(Hot_sn,'fro')^2 *100;
HotPrct3 = vecnorm(PERjja_sn'*HotV).^2/norm(PERjja_sn,'fro')^2 *100;
HotPrct4 = sum(HotU,1).*diag(HotS)'.*sum(HotV,1) /sum(HotCov(:))*100;
%ColdPrct1 = diag(ColdS).^2/norm(ColdCov,'fro')^2 *100;
%ColdPrct4 = sum(ColdU,1).*diag(ColdS)'.*sum(ColdV,1) /sum(ColdCov(:))*100;
clear HotCov ColdCov;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for n=1:4
subplot(3,4,n,'ActivePositionProperty','position');
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, reshape(HotU(:,n),ds(1:2))'); shading flat;
colormap(gca,b2r(-0.15,0.15)); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
%title({['Hot singular vector ',num2str(n)], ['square of singular value: ',num2str(HotPrct1(n),'%4.1f'),'%'], ['contrib. to scatter "r": ',num2str(HotPrct4(n),'%4.1f'),'%']},'fontsize',16);
%title({['Hot #',num2str(n),', ',num2str(HotPrct1(n),'%4.0f'),'%, ',num2str(HotPrct4(n),'%4.0f'),'%']},'fontsize',16);
title({[sprintf('Hot #%i,%.0f%%,%.0f%%,%.0f%%,%.0f%%',n,HotPrct1(n),HotPrct2(n),HotPrct3(n),HotPrct4(n))]},'fontsize',13);
tightmap;

subplot(3,4,4+n,'ActivePositionProperty','position');
subplot(3,4,4+n,'ActivePositionProperty','position');  % bug..
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, reshape(HotV(:,n),ds(1:2))'); shading flat;
colormap(gca,b2r(-0.15,0.15)); %colorbar;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
%title({['JJA blocking singular vector ',num2str(n)]},'fontsize',16);
title({['JJA blocking #',num2str(n)]},'fontsize',16);
%title({[titleArr{m},' #',num2str(n)]},'fontsize',16);
tightmap;

subplot(3,4,8+n,'ActivePositionProperty','position');
yyaxis left;
plot([yStart:yEnd], sum(HotU(:,n),1) *HotU(:,n)' *Hot_sn, '-o');
xlim([yStart yEnd]); xlabel({'year'});
ax=gca; ylim(max(abs(ax.YLim))*[-1 1]);
ax.TickLength = [0.05 0.05];
yyaxis right;
plot([yStart:yEnd], sum(HotV(:,n),1) *HotV(:,n)' *PERjja_sn, '-o');
ax=gca; ylim(max(abs(ax.YLim))*[-1 1]);
%if (contains(textTH, 'wgt'))
%  ylabel({'Mean weighted blocking area (unit*km^2)'});
%else
%  ylabel({'Mean blocking area (km^2)'});
%end

end %n loop
if (contains(textTH, 'wgt'))
  subplot(3,4,9); yyaxis left; ylabel({'Mean weighted extreme area (K*km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean weighted blocking area (unit*km^2)'});
else
  subplot(3,4,5); title({[titleArr{m},' #1']},'fontsize',12);
  subplot(3,4,9); yyaxis left; ylabel({'Mean extreme area (km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean blocking area (km^2)'});
end
print(gcf, '-dpsc2','-append',fn_figure);
  end  % m



%

end % textWgt loop

system(['ps2pdf ',fn_figure]);  % TODO
system(['rm ',fn_figure]);
%}

%% save 2 nc for ncl 20181010
%
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO

%vernumArr = {'5600','4002','8000','0602','2631' ,'4004','4014','x600'};  % qual
%vernumArr = {'5900','0a02','2937','4923','8900','5940','2946','4943','x900'};  % 170623
vernumArr = {'2a72','2a73','0a13','0a15'};  % 181010
%vernumArr = {'2az5','2a75','2az6','2a75'};  % 18 AGU
textWgtArr = {'chunk','lndchunk','lndchunk','lndchunk'};
%titleArr = {'DG83 prst=0.94 A=3: land and ocean','DG83 prst=0.94 A=3: land only','D13'
titleArr = {'DG83p: land and ocean','DG83p: land only','D13 on land: original threshold','D13 on land: tuned threshold'};
%titleArr = {'DG83: land and ocean','DG83: land only','',''};
%for textWgt = textWgtArr

%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH,'_',textWgt{1}];

clear bjjaArr e01arr e01;
  for m = 1:length(vernumArr)

textTH = textWgtArr{m};
ver = [vernumArr{m},'_',caseid];

if (contains(textTH, 'wgt'))
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','PERwjja_n');
  ydata = PERwjja_n;
else
  load(['../index_wise/BlockFreq_',verX,'_',caseid,'.mat'],'strTitle','PERjja_n');
  ydata = PERjja_n;
end
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rng default;
e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
e99 = (squeeze(mean(e99arr,1)));

switch textTH
 case 'chunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERjja_n');
  bjjaArr(:,m) = PERjja_n;
%  strT = 'JJA blocking area (km^2)';
 case 'lndchunk'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERljja_n');
  bjjaArr(:,m) = PERljja_n;
%  strT = 'JJA blocking area on land (km^2)';
 case 'chunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERwjja_n');
  bjjaArr(:,m) = PERwjja_n;
%  strT = 'Weighted JJA blocking area (unit*km^2)';
 case 'lndchunkwgt'
  load(['../index_wise/BlockFreq_',verX,ver,'.mat'],'strTitle','PERlwjja_n');
  bjjaArr(:,m) = PERlwjja_n;
%  strT = 'Weighted JJA blocking area on land (unit*km^2)';
 otherwise
  warning('unexpected textth')
end
rng default;
%e01(m) = (crossval('mse',bjjaArr(:,m),ydata,'Predfun',regf,'kfold',3,'mcreps',200))/e99;
e01arr(:,m) = crossval(errf,bjjaArr(:,m),ydata,'kfold',3,'mcreps',200);
  end  % m
e01 = (squeeze(mean(e01arr,1)))./e99;
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(ydata)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
sigjja = squeeze(cvtcdf(-e01arr+reshape(e01arr,[200*3 1 length(vernumArr)])))

fn_savenc = ['scatter_',verX,'_',caseid,'.nc'];
 system(['rm ',fn_savenc]);
nccreate(fn_savenc,'bjjaArr','Dimensions',{'t_jja',length(ydata),'index',length(vernumArr)},'DataType','single','Format','netcdf4')
nccreate(fn_savenc,'ydata','Dimensions',{'t_jja',length(ydata)},'DataType','single')
nccreate(fn_savenc,'e01','Dimensions',{'index',length(vernumArr)},'DataType','single')
nccreate(fn_savenc,'titleArr','Dimensions',{'charlen',size(char(titleArr)',1),'index',length(vernumArr)},'DataType','char')
nccreate(fn_savenc,'e99','DataType','single')
ncwrite(fn_savenc,'bjjaArr',bjjaArr)
ncwrite(fn_savenc,'ydata',ydata)
ncwrite(fn_savenc,'e01',e01)
ncwrite(fn_savenc,'titleArr',char(titleArr)')
ncwrite(fn_savenc,'e99',e99)
system(['ln -sf ',fn_savenc,' scatter.nc']);
system(['ncl -Q xy.grl_linefit.ncl']);

%% Block Freq (Pfahl2a in xtrm_colocate_pchan)
%{
% unweighted
vernumArr = {'4002','4004'};
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');

PERfreq_jja = mean(PER_jja,3);
PERfreq_djf = mean(PER_djf,3);
%PERfreq_jja(PERfreq_jja==0) = nan;
%PERfreq_djf(PERfreq_djf==0) = nan;
PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

 fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
fn_savenc = ['../index_wise/BlockFreq2_',textTH,'_',ver,'.nc'];
 system(['rm ',fn_savenc]);
system(['ncks -v lat,lat_bnds,lon,lon_bnds ',fn_t42,' ',fn_savenc]);
nccreate(fn_savenc,'PERfreq_jja','Dimensions',{'lon',128,'lat',64},'DataType','single','Format','classic')
nccreate(fn_savenc,'PERfreq_djf','Dimensions',{'lon',128,'lat',64},'DataType','single','Format','classic')
ncwrite(fn_savenc,'PERfreq_jja',PERfreq_jja)
ncwrite(fn_savenc,'PERfreq_djf',PERfreq_djf)
  end  % m
%}

%end % textWgt loop
%




%% 170404 Fig 4 for Qual
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd','latt42','lont42');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panel_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);

vernumArr = {'4002','4004'};
%spArr = [1,4];
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
textWgtArr = {'chunk'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];
%textTH = [textTH, '_chunk'];
%textTH = [textTH, '_chunkwgt'];

% Cold
%figure('units','inches','position',[0 1 8.5 8.5], 'paperUnits','inches','papersize',[8.5 8.5],'paperposition',[0 0 8.5 8.5]);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
spArr = [1,4];
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
subplot(2,3,spArr(m));
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
plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))],[min(bdjfArea_t(:)),max(bdjfArea_t(:))]*coldstat(1)+coldstat(2),'-','linewidth',2);
% http://www.originlab.com/doc/Origin-Help/LR-Algorithm   confidence ellipse
%inv22 = inv( cov(bdjfArea_t(:),coldArea_t(:)) );  % normalize by N-1
%xmean=mean(bdjfArea_t(:)); ymean=mean(coldArea_t(:));
%fcontour(@(x,y) [x-xmean y-ymean]*inv22*[x-xmean;y-ymean], [min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))],'r','linewidth',2,'LevelList',1);
%plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))], ([min(bdjfArea_t(:)),max(bdjfArea_t(:))]-xmean)*tan(atan2(-2*inv22(2),inv22(4)-inv22(1))/2)+ymean,'-','linewidth',2)

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
title({strTitle, ['cold, r=',num2str(coldstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(coldArea_t(:)),max(coldArea_t(:))]); axis square; %axis tight;

  end  % m

%% Block Freq (Pfahl2a in xtrm_colocate_pchan)
% unweighted
spArr = spArr+1;
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'],'strTitle');
subplot(2,3,spArr(m));

PERfreq_jja = mean(PER_jja,3);
PERfreq_djf = mean(PER_djf,3);
%PERfreq_jja(PERfreq_jja==0) = nan;
%PERfreq_djf(PERfreq_djf==0) = nan;
PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(jet(12)); caxis([0 24]); colorbar;  % fig2a in Scherrer 2006
plotm(coastlat,coastlon,'k')
%title('\fontsize{20}Relative frequency (%) of intense blocking events during DJF');
title({strTitle,'DJF blocking frequency (%)'},'fontsize',11);
tightmap;

  end  % m

%% plot polyfit: ENSO
% wget http://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
% awk 'BEGIN{FIELDWIDTHS = "5 5 7 7"} $1=="  DJF" && $2>=1980 && $2<=2015  {print $0}' oni.ascii.txt > oni_DJF.txt
oni_DJF = dlmread('oni_DJF.txt','',0,1);

spArr = spArr+1;
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
coldstat = [polyfit(bdjfArea_t(:),oni_DJF(:,3),1) corr(bdjfArea_t(:),oni_DJF(:,3))];
subplot(2,3,spArr(m));
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
  xlabel({'Weighted blocking area (unit*km^2)',strXCorr});
  ylabel({'Oceanic Ni$\tilde{n}$o Index',strYCorr});
else
  xlabel({'Blocking area (km^2)',strXCorr});
  ylabel({'Oceanic Ni$\tilde{n}$o Index',strYCorr},'interpreter','latex');
end
plot(bdjfArea_t(:),oni_DJF(:,3),'.','markersize',3);
plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))],[min(bdjfArea_t(:)),max(bdjfArea_t(:))]*coldstat(1)+coldstat(2),'-','linewidth',2);

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart+1:yEnd
    text(double(bdjfArea_t(yyyy-yStart)),double(oni_DJF(yyyy-yStart,3)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['ONI trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3)))]);
%  system(['echo cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3))), ' >> corrtrend-cold']);
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
%  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bdjfArea_t(:),oni_DJF(:,3));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({strTitle, ['r=',num2str(coldstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(oni_DJF(:,3)),max(oni_DJF(:,3))]); axis square; %axis tight;

  end  % m

%% SeasonalCycle, Area in xtrm_reanalysis_pchan02.m
% 'legend' only

print(gcf, '-dpsc2','-append',fn_figure);

end % textWgt loop

%system(['ps2pdf ',fn_figure]);
%}

%% 170408 fig4b
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd','latt42','lont42');

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['panel_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);

vernumArr = {'4002','4004','x600'};
%spArr = [1,4];
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
textWgtArr = {'chunk'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];
%textTH = [textTH, '_chunk'];
%textTH = [textTH, '_chunkwgt'];

% Cold
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);

%% plot polyfit: ENSO
% wget http://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
% awk 'BEGIN{FIELDWIDTHS = "5 5 7 7"} $1=="  DJF" && $2>=1980 && $2<=2015  {print $0}' oni.ascii.txt > oni_DJF.txt
oni_DJF = dlmread('oni_DJF.txt','',0,1);

spArr = [1,2,3];
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
coldstat = [polyfit(bdjfArea_t(:),oni_DJF(:,3),1) corr(bdjfArea_t(:),oni_DJF(:,3))];
subplot(2,3,spArr(m));
hold on;
if (contains(textTH, 'chunk'))
  strXCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)),'%+.3f'),')'];
  strYCorr = ['(r value for trend = ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3)),'%+.3f'),')'];
else
  strXCorr=[]; strYCorr=[];
end
if (contains(textTH, 'wgt'))
  xlabel({'Weighted blocking area (unit*km^2)',strXCorr});
  ylabel({'Oceanic Ni$\tilde{n}$o Index',strYCorr});
else
  xlabel({'Blocking area (km^2)',strXCorr});
  ylabel({'Oceanic Ni$\tilde{n}$o Index',strYCorr},'interpreter','latex');
end
plot(bdjfArea_t(:),oni_DJF(:,3),'.','markersize',3);
plot([min(bdjfArea_t(:)),max(bdjfArea_t(:))],[min(bdjfArea_t(:)),max(bdjfArea_t(:))]*coldstat(1)+coldstat(2),'-','linewidth',2);

if (contains(textTH, 'chunk'))
  mytext = text;
  clear text;
  for yyyy = yStart+1:yEnd
    text(double(bdjfArea_t(yyyy-yStart)),double(oni_DJF(yyyy-yStart,3)), sprintf('%02d',mod(yyyy,100)) );
  end
  text = mytext;
  disp(['ONI trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3)))]);
%  system(['echo cold trend ',textTH,'_',caseid,':  ',num2str(corr((yStart+1:yEnd)',oni_DJF(:,3))), ' >> corrtrend-cold']);
  disp(['blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:)))]);
%  system(['echo blockDJF trend ',textTH,'_',ver,':  ',num2str(corr((yStart+1:yEnd)',bdjfArea_t(:))), ' >> corrtrend-cold']);
else
  [Ncounts,Xedges,Yedges] = histcounts2(bdjfArea_t(:),oni_DJF(:,3));
  Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
  Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
  contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.40','k');  % TODO
%  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.40',3,'k');
end
title({strTitle, ['r=',num2str(coldstat(3),'%+.3f')]}, 'interpreter','none');
axis([min(bdjfArea_t(:)),max(bdjfArea_t(:)), min(oni_DJF(:,3)),max(oni_DJF(:,3))]); axis square; %axis tight;

  end  % m
subplot(2,3,3);
if (contains(textTH, 'wgt'))
  xlabel({'Weighted extreme area (K*km^2)',strXCorr});
else
  xlabel({'Extreme area (km^2)',strXCorr});
end

%% SeasonalCycle, Area in xtrm_reanalysis_pchan02.m
% 'legend' only

print(gcf, '-dpsc2','-append',fn_figure);

end % textWgt loop

%system(['ps2pdf ',fn_figure]);
%}

%% 170404 Fig 6 for Qual
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'yStart','yEnd');

fn_figure = ['panel_',textTH,'_',text,'.ps'];
%system(['rm ',fn_figure]);

vernumArr = {'4002','4004','4014'};
%textWgtArr = {'chunk','chunkwgt','org','wgt'};
textWgtArr = {'chunk'};
for textWgt = textWgtArr

%thresh{5} = 0;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];
%textTH = [textTH, '_chunk'];
%textTH = [textTH, '_chunkwgt'];

% Hot
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);
subplot(2,3,m);
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
plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))],[min(bjjaArea_t(:)),max(bjjaArea_t(:))]*hotstat(1)+hotstat(2),'-','linewidth',2);
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

  end  % m


print(gcf, '-dpsc2','-append',fn_figure);

end % textWgt loop

system(['ps2pdf ',fn_figure]);
%}

%% save 2 nc for ncl
%{
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;  % TODO

%vernumArr = {'5600','4002','8000','0602','2631' ,'4004','4014','x600'};  % qual
vernumArr = {'5900','0a02','2937','4923','8900','5940','2946','4943','x900'};  % 170623
textWgtArr = {'chunk'};
for textWgt = textWgtArr

textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
textTH = [textTH,'_',textWgt{1}];

  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
  load(['../index_wise/scatter_',textTH,'_',ver,'.mat']);

fn_savenc = ['../index_wise/scatter_',textTH,'_',ver,'.nc'];
 system(['rm ',fn_savenc]);
nccreate(fn_savenc,'bjjaArea_t','Dimensions',{'t_jja',length(bjjaArea_t)},'DataType','single','Format','classic')
nccreate(fn_savenc,'hotArea_t','Dimensions',{'t_jja',length(bjjaArea_t)},'DataType','single')
nccreate(fn_savenc,'hotstat','Dimensions',{'stat',3},'DataType','single')
nccreate(fn_savenc,'bdjfArea_t','Dimensions',{'t_djf',length(bdjfArea_t)},'DataType','single')
nccreate(fn_savenc,'coldArea_t','Dimensions',{'t_djf',length(bdjfArea_t)},'DataType','single')
nccreate(fn_savenc,'coldstat','Dimensions',{'stat',3},'DataType','single')
ncwrite(fn_savenc,'bjjaArea_t',bjjaArea_t)
ncwrite(fn_savenc,'hotArea_t',hotArea_t)
ncwrite(fn_savenc,'hotstat',hotstat)
ncwrite(fn_savenc,'bdjfArea_t',bdjfArea_t)
ncwrite(fn_savenc,'coldArea_t',coldArea_t)
ncwrite(fn_savenc,'coldstat',coldstat)
  end  % m

%% Block Freq (Pfahl2a in xtrm_colocate_pchan)
%{
% unweighted
vernumArr = {'4002','4004'};
  for m = 1:length(vernumArr)

ver = [vernumArr{m},'_',caseid];
load(['../index_wise/BlockFreq_',ver,'.mat'],'prm','PER_jja','PER_djf','Wgt_jja','Wgt_djf');

PERfreq_jja = mean(PER_jja,3);
PERfreq_djf = mean(PER_djf,3);
%PERfreq_jja(PERfreq_jja==0) = nan;
%PERfreq_djf(PERfreq_djf==0) = nan;
PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;
PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

 fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
fn_savenc = ['../index_wise/BlockFreq2_',textTH,'_',ver,'.nc'];
 system(['rm ',fn_savenc]);
system(['ncks -v lat,lat_bnds,lon,lon_bnds ',fn_t42,' ',fn_savenc]);
nccreate(fn_savenc,'PERfreq_jja','Dimensions',{'lon',128,'lat',64},'DataType','single','Format','classic')
nccreate(fn_savenc,'PERfreq_djf','Dimensions',{'lon',128,'lat',64},'DataType','single','Format','classic')
ncwrite(fn_savenc,'PERfreq_jja',PERfreq_jja)
ncwrite(fn_savenc,'PERfreq_djf',PERfreq_djf)
  end  % m
%}

end % textWgt loop
%}




% outdated: Pfahl plot scatter
%{
[Ncounts,Xedges,Yedges] = histcounts2(bjjaArea_t(:),hotArea_t(:));
Xedges = (Xedges(2:end)+Xedges(1:end-1))/2;
Yedges = (Yedges(2:end)+Yedges(1:end-1))/2;
%subplot(2,3,1)
hold on;
xlabel({'','Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
plot(bjjaArea_t(:),hotArea_t(:),'.','markersize',3);
contour( Xedges(1:end),Yedges,Ncounts(1:end,:).^0.50','k');
%contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.30',3,'k');
plot([min(bjjaArea_t(:)),max(bjjaArea_t(:))],[min(bjjaArea_t(:)),max(bjjaArea_t(:))]*hotstat(1)+hotstat(2),'-','linewidth',2);
title({'Pfahl and Wernli 2012 index', ['hot, r=',num2str(hotstat(3),'%+.3f')]}, 'interpreter','none');

pause(5);
savefig(gcf,['scatter_',textTH,'_',caseid,'.fig'])
print(gcf,'-dpdf',['scatter_',textTH,'_',caseid,'.pdf'])
%}

% SeasonalCycle, Area
%{
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);

thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
load(['temp_',textTH,'_',text,'.mat'], 'nyr',  'yStart','yEnd')

%textTH = [textTH, '_org'];
textTH = [textTH, '_wgt'];

ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,1); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
%xlabel({'Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
xlabel({'day'}); ylabel('Extreme area (km^2)');
plot(reshape( hotArea_t,[],nyr ));
%plot([1 92],HotQuantile(i,j)*[1 1],'k-');
title(['JJA']);

subplot(3,4,2); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
%ylabel('year');
%plot([1;1]*[yStart:yEnd]);
xlabel({'year'}); ylabel('Mean extreme area (km^2)');
plot([1;1]*[yStart:yEnd], [zeros(1,nyr); mean( reshape(hotArea_t(:),[],nyr ), 1)]);
title(['Legend']);
xlim([yStart yEnd]);

ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,5); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bjjaArea_t,[],nyr ));
title('Dunn-Sigouin and Son 2013');

ver=['2631_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,6); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bjjaArea_t,[],nyr ));
title('Hassanzadeh et al. 2014');

ver=['4001_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,9); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bjjaArea_t,[],nyr ));
title('Scherrer et al. 2006');

ver=['7050_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,10); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bjjaArea_t,[],nyr ));
title('v850 quantile (7050)');


ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,4); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
%xlabel({'Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
xlabel({'day'}); ylabel('Extreme area (km^2)');
plot(reshape( coldArea_t,[],nyr-1 ));
%plot([1 92],ColdQuantile(i,j)*[1 1],'k-');
title(['DJF']);

ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,7); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bdjfArea_t,[],nyr-1 ));
title('Dunn-Sigouin and Son 2013');

ver=['2631_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,8); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bdjfArea_t,[],nyr-1 ));
title('Hassanzadeh et al. 2014');

ver=['4001_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,11); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bdjfArea_t,[],nyr-1 ));
title('Scherrer et al. 2006');

ver=['7050_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,12); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
xlabel({'day'}); ylabel('Blocked area (km^2)');
plot(reshape( bdjfArea_t,[],nyr-1 ));
title('v850 quantile (7050)');

savefig(gcf,['SeasonalCycleJJAareas_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['SeasonalCycleJJAareas_',textTH,'_',text,'.pdf'])
%}

%% case study 20170929 newer than 07.m
%{
clearvars -except  verX caseid;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
%caseid=['ERA-interim_19790101-20151231'];
%verX='x909';
load(['temp_',verX,'_',caseid,'.mat']);
disp('finish load'); toc

%vernumArr = {'2a71','2a68','2a69','0a13','4a26','8a14','9a00',verX};  % 180627
%titleArr = {'DG83,prst=.9375,A=2.5','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA','Extreme'};
vernumArr = {'0a15','0a13',verX};  % 181015
titleArr = {'tuned D13','D13','Extreme'};

fn_z500a  = ['../index_wise/Var_',verX,'0axx_',caseid,'.mat'];
mat_z500a = matfile(fn_z500a);
%Z500a = mat_z500a.ZaDaily;  % already in latt42/lat0
Z500 = mat_z500a.VarDaily;  %already in latt42/lat0
Z500a_jja = mat_z500a.Wgt_jja;
%Z500a_djf = mat_z500a.Wgt_djf;

%{
%% load nc, check lat lon
nc_z850 = ['../ERA-interim/z850_00Z_',caseid,'.nc'];
Z850 = ncread(nc_z850,'z')/9.81;
lattmp = ncread(nc_z850,'latitude');
lontmp = ncread(nc_z850,'longitude');
if (max(abs(latt42-lattmp))>0.1)
  Z850 = Z850(:,end:-1:1 ,:);
  lattmp = lattmp(end:-1:1);
end
if (max(abs(latt42-lattmp))>0.1 || max(abs(lont42-lontmp))>0.1)
  disp('error')
end

fn_tend = ['../ncl/daily6h_',caseid,'.nc'];
T850  = ncread(fn_tend,'T850Daily');
W850  = ncread(fn_tend,'W850Daily');
T850z = ncread(fn_tend,'S850Daily');
T850AdvU = ncread(fn_tend,'T850AdvU');
T850AdvV = ncread(fn_tend,'T850AdvV');
T850AdvW = ncread(fn_tend,'T850AdvW');
T850err  = ncread(fn_tend,'T850err');
latncl = ncread(fn_tend,'lat');
lonncl = ncread(fn_tend,'lon');
if (max(abs(latt42-latncl))>0.1)
  T850 = T850(:,end:-1:1 ,:);
  W850 = W850(:,end:-1:1 ,:);
  T850z = T850z(:,end:-1:1 ,:);
  T850AdvU = T850AdvU(:,end:-1:1 ,:);
  T850AdvV = T850AdvV(:,end:-1:1 ,:);
  T850AdvW = T850AdvW(:,end:-1:1 ,:);
  T850err = T850err(:,end:-1:1 ,:);
  latncl = latncl(end:-1:1);
end
if (max(abs(latt42-latncl))>0.1 || max(abs(lont42-lonncl))>0.1)
  disp('error')
end
%}

% PER
PERm_jja = zeros([ds_jja,numel(vernumArr)],'single');
  for m = 1:length(vernumArr)
ver = [vernumArr{m},'_',caseid];
mat_load2 = matfile(['../index_wise/BlockFreq_',verX,ver,'.mat']);
%PERm_jja(:,:,:,m) = mat_load2.PER_jja;
PERm_jja(:,:,:,m) = mat_load2.PERid_jja;
if (m==1) sdJJA_NH = mat_load2.sdJJA_NH; end
  end  % m
PERm_jja(PERm_jja==0) = nan;
%Hotid_jja(Hotid_jja==0) = nan;

%{
ver=['2a53_',caseid];
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
load(fn_load2,'PERid_jja','PERjjaAttr');
load(fn_load2,'sdJJA_NH');  %2a53

ver=['7050_',caseid];  %
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
 %load(fn_load2,'timeNan')
PER7050 = (mat_load2.PER);
Wgt7050 = mat_load2.T850Adv3d;

ver=['2631_',caseid]; u=2;v=1;  %Hybrid2020, zprime02, bugfix+4
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
PESR2631 = (squeeze(mat_load2.PESR(:,:,:,u)));
E2631 = (squeeze(mat_load2.E(:,:,:,u)));
%}

%% nan, sign, angle, demean, movmean
%{
%T850 = movmean(T850, prm.D,3);  %TODO necessary??
%timeNan = unique([timeNan, 1:(prm.D-1)/2,ds(3)-(prm.D-1)/2+1:ds(3)]);

% 15/15day*11yr
%
T850_star = movmean(T850, 15,3);  % no Endpoints treatment for star..
T850_star = movmean(T850_star, 15,3);  % no Endpoints treatment for star..
W850_star = movmean(W850, 15,3);  % no Endpoints treatment for star..
W850_star = movmean(W850_star, 15,3);  % no Endpoints treatment for star..
T850AdvU_star = movmean(T850AdvU, 15,3);  % no Endpoints treatment for star..
T850AdvU_star = movmean(T850AdvU_star, 15,3);  % no Endpoints treatment for star..
T850AdvV_star = movmean(T850AdvV, 15,3);  % no Endpoints treatment for star..
T850AdvV_star = movmean(T850AdvV_star, 15,3);  % no Endpoints treatment for star..
T850AdvW_star = movmean(T850AdvW, 15,3);  % no Endpoints treatment for star..
T850AdvW_star = movmean(T850AdvW_star, 15,3);  % no Endpoints treatment for star..
T850err_star = movmean(T850err, 15,3);  % no Endpoints treatment for star..
T850err_star = movmean(T850err_star, 15,3);  % no Endpoints treatment for star..

T850_hat = nan([ds(1:2) 366 nyr],'single');
W850_hat = nan([ds(1:2) 366 nyr],'single');
T850AdvU_hat = nan([ds(1:2) 366 nyr],'single');
T850AdvV_hat = nan([ds(1:2) 366 nyr],'single');
T850AdvW_hat = nan([ds(1:2) 366 nyr],'single');
T850err_hat = nan([ds(1:2) 366 nyr],'single');

for t = 1:366
  tArr = days( datetime('0000-01-01')+caldays(t-1)+calyears(yStart:yEnd) - f_h2d(time(1)) )+1; % order of addition matters
  T850_hat(:,:,t,:) = T850_star(:,:,tArr);
  W850_hat(:,:,t,:) = W850_star(:,:,tArr);
  T850AdvU_hat(:,:,t,:) = T850AdvU_star(:,:,tArr);
  T850AdvV_hat(:,:,t,:) = T850AdvV_star(:,:,tArr);
  T850AdvW_hat(:,:,t,:) = T850AdvW_star(:,:,tArr);
  T850err_hat(:,:,t,:) = T850err_star(:,:,tArr);
end
%T850_hat(:,:,[1:104,end-75:end]) = nan;  % jump elsewhere not Jan 1..
T850_hat = movmean(T850_hat,11,4,'omitnan');
W850_hat = movmean(W850_hat,11,4,'omitnan');
T850AdvU_hat = movmean(T850AdvU_hat,11,4,'omitnan');
T850AdvV_hat = movmean(T850AdvV_hat,11,4,'omitnan');
T850AdvW_hat = movmean(T850AdvW_hat,11,4,'omitnan');
T850err_hat = movmean(T850err_hat,11,4,'omitnan');

dtArr = f_h2d(time); DArr = 366*(dtArr.Year-yStart);
dtArr.Year=0; DArr = DArr + days(dtArr - datetime('0000-01-01') )+1;
%T850 = T850 - T850_bar - T850_hat(:,:,DArr);  % prime
T850 = T850 - T850_hat(:,:,DArr);  % prime
W850 = W850 - W850_hat(:,:,DArr);  % prime
T850AdvU = T850AdvU - T850AdvU_hat(:,:,DArr);  % prime
T850AdvV = T850AdvV - T850AdvV_hat(:,:,DArr);  % prime
T850AdvW = T850AdvW - T850AdvW_hat(:,:,DArr);  % prime
T850err = T850err - T850err_hat(:,:,DArr);  % prime
%T850Cli = T850_hat(:,:,DArr);%T850_bar + 
clear T850_bar T850_star T850_hat tArr dtArr DArr
clear W850_bar W850_star W850_hat
clear T850AdvU_bar T850AdvU_star T850AdvU_hat
clear T850AdvV_bar T850AdvV_star T850AdvV_hat
clear T850AdvW_bar T850AdvW_star T850AdvW_hat
clear T850err_bar T850err_star T850err_hat
%}

%% collect JJA
timeNan = [];

Z500_jja = zeros(ds_jja,'single');
%Z850_jja = zeros(ds_jja,'single');
%T850_jja = zeros(ds_jja,'single');
%W850_jja = zeros(ds_jja,'single');
%T850z_jja = zeros(ds_jja,'single');
%T850AdvU_jja = zeros(ds_jja,'single');
%T850AdvV_jja = zeros(ds_jja,'single');
%T850AdvW_jja = zeros(ds_jja,'single');
%T850err_jja = zeros(ds_jja,'single');
%PER7050_jja = false(ds_jja);
%Wgt7050_jja = zeros(ds_jja,'single');
%PESR2631_jja = zeros(ds_jja,'single');
%E2631_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  Z500_jja(:,:,tpointer+(0:tend-tstart)) = Z500(:,:,tstart:tend);
%  Z850_jja(:,:,tpointer+(0:tend-tstart)) = Z850(:,:,tstart:tend);
%  T850_jja(:,:,tpointer+(0:tend-tstart)) = T850(:,:,tstart:tend);
%  W850_jja(:,:,tpointer+(0:tend-tstart)) = W850(:,:,tstart:tend);
%  T850z_jja(:,:,tpointer+(0:tend-tstart)) = T850z(:,:,tstart:tend);
%  T850AdvU_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvU(:,:,tstart:tend);
%  T850AdvV_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvV(:,:,tstart:tend);
%  T850AdvW_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvW(:,:,tstart:tend);
%  T850err_jja(:,:,tpointer+(0:tend-tstart)) = T850err(:,:,tstart:tend);
%  PER7050_jja(:,:,tpointer+(0:tend-tstart)) = PER7050(:,:,tstart:tend);
%  Wgt7050_jja(:,:,tpointer+(0:tend-tstart)) = Wgt7050(:,:,tstart:tend);
%  PESR2631_jja(:,:,tpointer+(0:tend-tstart)) = PESR2631(:,:,tstart:tend);
%  E2631_jja(:,:,tpointer+(0:tend-tstart)) = E2631(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
clear PER7050 Wgt7050 Z500 Z850 PESR2631 E2631  T850 W850 T850z T850AdvU T850AdvV T850AdvW T850err

 %TODO some smoothing
 % ~/script/blocking/tech_note_Dunn-Sigouin_and_Son.txt
 %Reversal    GHG.F
 %        find (I,J) of max zprime. remove if lat >80N, <22.5N
 %        require any one (i,j) in (I-2:I+2, J-3:J+3), s.t. z(i,j)>z(i,j-5)
R_jja = zeros(ds_jja,'single');
R_jja(:,[yN1:yN2],:) = Z500_jja(:,[yN1:yN2],:) - Z500_jja(:,[yN1:yN2]-5,:);

%% remove trend
%{
  wrk  = nanmean( double(Z850_jja),3); % double precision needed?
  Z850_jja = Z850_jja - repmat( wrk, [1 1 ds_jja(3)]);

% 90day*5yr
%{
T850jja_xyn = squeeze(mean( reshape(T850_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%W850jja_xyn = squeeze(mean( reshape(W850_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850AdvUjja_xyn = squeeze(mean( reshape(T850AdvU_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850AdvVjja_xyn = squeeze(mean( reshape(T850AdvV_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850AdvWjja_xyn = squeeze(mean( reshape(T850AdvW_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
T850errjja_xyn = squeeze(mean( reshape(T850err_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr

T850_jja = T850_jja - reshape(repmat(reshape(movmean(T850jja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
%W850_jja = W850_jja - reshape(repmat(reshape(movmean(W850jja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850AdvU_jja = T850AdvU_jja - reshape(repmat(reshape(movmean(T850AdvUjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850AdvV_jja = T850AdvV_jja - reshape(repmat(reshape(movmean(T850AdvVjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850AdvW_jja = T850AdvW_jja - reshape(repmat(reshape(movmean(T850AdvWjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
T850err_jja = T850err_jja - reshape(repmat(reshape(movmean(T850errjja_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
%}

% no quantile
      
% no land

T850tend_jja = T850AdvU_jja+T850AdvV_jja+T850AdvW_jja+T850err_jja;
%}

%{
ind = find( time_jja == hours(datetime(2010,7,28,0,0,0) -datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S')) ) +(-4:5);
%ind = 3359 +(-1:1);

F_y      = repmat(reshape(1:floor(ds(2)/2), [floor(ds(2)/2) 1]), [1 ds(1)]);
%F_y      = repmat(reshape(1:ds(2), [ds(2) 1]), [1 ds(1)]);
%F_y_yxh  = zeros([floor(ds(2)/2) ds(1) 2],'single');
%F_y_yxh(:,:,1) = F_y(1:floor(ds(2)/2)           ,:);
%F_y_yxh(:,:,2) = F_y(end:-1:end-floor(ds(2)/2)+1,:);

F_repmat = repmat(reshape(1:ds(1), [ds(1) 1]), [1 ds(2)]);
F_sin    = sin( F_repmat*2*pi/ds(1)) .*repmat(areaEarth(:)', [ds(1) 1]);
F_cos    = cos( F_repmat*2*pi/ds(1)) .*repmat(areaEarth(:)', [ds(1) 1]);

nBlock = 0; xyBlock =[];
TBlock=0;
S_sin =0; S_cos =0;

for iii = ind(:)'  % row vector
    t = iii;
%    [mh t] = ind2sub(size(bArea_ht), iii);
    if (any(any(Hot_jja(:,:,iii) & repmat(areaEarth(:)'>0, [ds(1) 1]) )))

%     idUniq = unique(Hot_jja(:,:,iii));
%     for id = idUniq(idUniq>0).'
%       % for cyclic boundary condition, find xRef (labels dont cross) to be boundary
%       xRef = find( all(F_id(:,:,t)~=id, 2), 1);
%       if (isempty(xRef))
%           disp(['t=',num2str(t), ', id=',num2str(id)])
%           continue
%       end
%       F_x = xRef + mod(F_repmat-xRef, ds(1));
%       tmp = [t; id; mean(F_x(F_id(:,:,t)==id)); mean(F_y(F_id(:,:,t)==id))];
%         % un-weighted average of x,y index

        S_sin = S_sin +sum(F_sin(Hot_jja(:,:,iii)));
        S_cos = S_cos +sum(F_cos(Hot_jja(:,:,iii)));
        tmp(3) = atan2( sum(F_sin(Hot_jja(:,:,iii))),sum(F_cos(Hot_jja(:,:,iii))) )/2/pi*ds(1);
        tmp = [t; 0; tmp(3); (0)];
%        tmp = [t; 0; tmp(3); ( (ds(2)+1)*(mh==2) +(-1)^(mh==2) *mean(F_y(Hot_jja(:,:,iii))) )];
%        tmp = [t; 0; tmp(3); mean(F_y_yxh( Hot_jja(:,:,iii) ,mh))];
          % un-weighted average of x,y index
        tmp(3) = 0.5 + mod(tmp(3)-0.5, ds(1));  % range from 0.5 - ds(1).5
        xyBlock = [xyBlock tmp];
    end
end

        tmp(3) = atan2( S_sin,S_cos )/2/pi*ds(1);
        tmp(3) = 0.5 + mod(tmp(3)-0.5, ds(1));  % range from 0.5 - ds(1).5

  tmp(3) = 40/360*ds(1);
%}

yyyy = 2010;
 %strcmp(HotAttr.id,'2308357')
%m=1; csort=594;
%xxx Multid_jja = removecats(Hotid_jja.*PERid_jja);
%xxx ids = str2double(split(categories(Multid_jja)));
%xxx numPER_Hot = countcats(categorical(ids(:,1),str2double(HotAttr.id)));
%xxx areaHot_PER = zeros([numel(PERjjaAttr.id) 1],'single');
%xxx for m = 1:size(ids,1)
%xxx   areaHot_PER(ids(m,2)) = areaHot_PER(ids(m,2))+HotAttr.area(ids(m,1))/numPER_Hot(ids(m,1));
%xxx end
%numHot_PER = countcats(categorical(ids(:,2),str2double(PERjjaAttr.id)));

%csort = find(numPER_Hot==2);  %split
%[~,csort] = sort(HotAttr.area.*(~numPER_Hot),'descend'); csort=csort(1:10);  % missed event
%[~,csort] = sort(PERjjaAttr.arealsm.*(~numHot_PER),'descend'); csort=csort(1:10);  %False alarm
%csort = [169;147]; %Weak (False alarm)
%[~,csort] = sort(areaHot_PER,'descend'); csort=csort(1:5);  %Strong case
%csort = [171;89]; %Good Catch
%[~,csort] = sort(PERjjaAttr.arealsm.*(ceil(PERjjaAttr.tstart/nd_jja)==(1991-yStart+1)),'descend'); csort=csort(1:6);  %1991
%[~,csort] = sort(HotAttr.area.*(ceil(HotAttr.tstart/nd_jja)==(1991-yStart+1)),'descend'); csort=csort(1:10);  %1991
%disp('pause'); pause;
%csort = [445;453;454]; %1991

%tstart = HotAttr.tstart(csort); tend = HotAttr.tend(csort);
%xxx x = lont42(round(HotAttr.x(csort))); y = latt42(round(HotAttr.y(csort)));
 % :%s/latt42(round(HotAttr\.y(csort(m))))/y(m)/gce | %s/lont42(round(HotAttr\.x(csort(m))))/x(m)/gce
%ind = find( time_jja == hours(datetime(1995,7,12,0,0,0) -datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S')) ) +(-1:5); y=42; x=-88; csort=csort(2);
ind = find( time_jja == hours(datetime(2010,7,29,0,0,0) -datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S')) ) +(-5:5); y=57; x=45; nCase=1; csort=1;%csort(2);

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added
%rng(2017);

%fn_figure = ['casemissjja',verX,ver(1:4),'.ps'];
%fn_figure = ['case',datestr( f_h2d(mean(time_jja(ind))),'yyyymmdd'),'.ps'];
fn_figure = ['case',verX,'_',num2str(yyyy),'_D13.ps'];
system(['rm ',fn_figure]);

% area time series
%
figure('units','inches','position',[0 1 12 9], 'paperUnits','inches','papersize',[12 9],'paperposition',[0 0 12 9]);

%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];

%wrk = (hours(mean(time_jja(ind)))+datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
%yyyy = wrk.Year;
%yyyy = ceil(mean(ind)/nd_jja);  % index starting 1
%wrk = (yyyy-1)*nd_jja +(1:nd_jja);
wrk=(yyyy-yStart)*nd_jja+(1:nd_jja);
%figure; plot(wrk,bjjaArea_t(wrk),'k-', wrk,hotArea_t(wrk),'r-');
%plot(mod(ind,nd_jja),bjjaArea_t(ind));

nr=1; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=346|lont42<=80,:,nr)=true; regArr{nr}='land on 14W-80E';
ydata = squeeze(sum(areaEarth .* (mean((PERm_jja(:,:,wrk,:)>0).*lsm_jja.*mask_xyr(:,:,nr),1)),2));

subplot(3,1,1:2); hold on;
%ax=gca; ax.ColorOrder = jet(nyr); ax.ColorOrderIndex=yyyy;
ylabel('Area (km^2)'); %xlabel({'day'});
%plot([1:nd_jja],ydata);
plot(datenum(f_h2d(time_jja(wrk))),ydata); datetick('x','mm/dd'); grid on;
%plot([1 nd_jja],HotQuantile(i,j)*[1 1],'k-');
title([regArr{nr},', ',num2str(yyyy)]);
legend(titleArr,'location','northwest');
set(gca,'FontSize',20);

%{
ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,1); hold on;
ax=gca; ax.ColorOrder = jet(nyr); ax.ColorOrderIndex=yyyy;
%xlabel({'Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
ylabel('Extreme area (km^2)'); %xlabel({'day'});
%plot(reshape( hotArea_t,[],nyr ));
%plot([1:nd_jja],hotArea_t(wrk));
plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
%plot([1 nd_jja],HotQuantile(i,j)*[1 1],'k-');
title([num2str(yyyy+yStart-1)]);

ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,5); hold on;
ax=gca; ax.ColorOrder = jet(nyr); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bjjaArea_t,[],nyr ));
%plot([1:nd_jja],bjjaArea_t(wrk));
plot(datenum(f_h2d(time_jja(wrk))),bjjaArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('Dunn-Sigouin and Son 2013');

ver=['2631_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,6); hold on;
ax=gca; ax.ColorOrder = jet(nyr); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bjjaArea_t,[],nyr ));
%plot([1:nd_jja],bjjaArea_t(wrk));
plot(datenum(f_h2d(time_jja(wrk))),bjjaArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('Hassanzadeh et al. 2014');

ver=['4001_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,9); hold on;
ax=gca; ax.ColorOrder = jet(nyr); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bjjaArea_t,[],nyr ));
%plot([1:nd_jja],bjjaArea_t(wrk));
plot(datenum(f_h2d(time_jja(wrk))),bjjaArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('Scherrer et al. 2006');

ver=['7050_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,10); hold on;
ax=gca; ax.ColorOrder = jet(nyr); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bjjaArea_t,[],nyr ));
%plot([1:nd_jja],bjjaArea_t(wrk));
plot(datenum(f_h2d(time_jja(wrk))),bjjaArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('v850 quantile (7050)');
%}

%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
print(gcf, '-dpsc2','-append',fn_figure);
%

%% map
%
disp('start map'); toc
figure('units','inches','position',[0 1 12 9], 'paperUnits','inches','papersize',[12 9],'paperposition',[0 0 12 9]);
%figure('Position',[0 80 1920 1000],'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%figure('OuterPosition',[0 80 1366 700]);
%'meridianlabel','on','parallellabel','on','mlinelocation',60,'plinelocation',30,

%for iii = ind(:)'  % row vector
 %   [mh t] = ind2sub(size(bArea_ht), iii);
 %   if (any(any(Hot_jja(:,:,iii) & repmat(areaEarth(:)'>0, [ds(1) 1]) )))
%xxx for m = 1:numel(csort)
 %  for iii = HotAttr.tstart(csort(m))-2:HotAttr.tend(csort(m))+2
for iii = [1:nd_jja]+(2010-yStart)*nd_jja
%xxx tstart = min([HotAttr.tstart(csort(m));PERjjaAttr.tstart( ids(ids(:,1)==csort(m),2) )]);
%xxx tend   = max([HotAttr.tend(csort(m));  PERjjaAttr.tend( ids(ids(:,1)==csort(m),2) )]);
%tstart = min([PERjjaAttr.tstart(csort(m));HotAttr.tstart( ids(ids(:,2)==csort(m),1) )]);
%tend   = max([PERjjaAttr.tend(csort(m));  HotAttr.tend( ids(ids(:,2)==csort(m),1) )]);
%xxx for iii = tstart-1:tend+1
 %HotAttr.tmax(csort(m))
clf;

%pcolor loop
%
  for m = 1:length(vernumArr)
f_tp = @(a,b,c) ceil(c/a) +b*mod(c-1,a);
subplot(2,2,m); hold off;
%subplot(2,4,f_tp(2,4,m));
%axesm('MapProjection','pcarree','MapLatLimit',y(nCase)+[-25 25],'MapLonLimit',x(nCase)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
axesm('MapProjection','eqaazim','origin',[y(nCase) x(nCase)],'FLatLimit',[-Inf 50],'grid','on','mlinelocation',20,'plinelocation',20);
%contourm(latt42,lonplot,double(Hotid_jja([1:end,1],:,iii))',[0.5 6.5]); axis equal tight;
pcolormPC(latt42,lont42,PERm_jja(:,:,iii,m)');
hold on;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
colormap(gca,jet); caxis([csort(nCase)-10 csort(nCase)+5]); colorbar; caxis auto;

%contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-');
title({[titleArr{m},' ',datestr( f_h2d(time_jja(iii)) )]});
tightmap;
  end  % m
%

%pcolor id
%{
subplot(2,2,1); hold off;
%axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
axesm('MapProjection','eqaazim','origin',[y(m) x(m)],'FLatLimit',[-Inf 50],'grid','on','mlinelocation',20,'plinelocation',20);
%contourm(latt42,lonplot,double(Hotid_jja([1:end,1],:,iii))',[0.5 6.5]); axis equal tight;
pcolormPC(latt42,lont42,Hotid_jja(:,:,iii)');
hold on;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
colormap(gca,jet); caxis([csort(m)-10 csort(m)+5]); colorbar; %caxis auto;

%contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-');
%title({'Hotid (color)'});
title({['Hotid (color) ',datestr( f_h2d(time_jja(iii)) )]});
%title({['Hotid (color) ','iii=',num2str(iii)]});
tightmap;
%

subplot(2,2,3); hold off;
%axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
axesm('MapProjection','eqaazim','origin',[y(m) x(m)],'FLatLimit',[-Inf 50],'grid','on','mlinelocation',20,'plinelocation',20);
%contourm(latt42,lonplot,double(PERid_jja([1:end,1],:,iii))',[0.5 6.5]); axis equal tight;
pcolormPC(latt42,lont42,PERid_jja(:,:,iii)');
hold on;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
colormap(gca,jet); caxis([csort(m)-5 csort(m)+5]); colorbar; caxis auto;

%contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-');
title({'PERid (color)'});
%title({['PERid (color) ',datestr( f_h2d(time_jja(iii)) )]});
tightmap;
%}

%title
%{
subplot(3,3,1); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(mx2t_jja([1:end,1],:,iii))',[4:2:10],'linewidth',2); axis equal tight;
colormap(gca,jet); caxis([4 10]); %colorbar;
title({['mx2t (color) ','iii=',num2str(iii)]});
%title({['mx2t (color) ',datestr( f_h2d(time_jja(iii)) )]});
tightmap;
%}

%template
%{
subplot(3,3,4); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(Z500_jja([1:end,1],:,iii))',[5000:100:5800],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'Z500 (color)'});
tightmap;

subplot(3,3,7); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[50:50:200],'r-','linewidth',2); axis equal tight;
contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-100:50:-50],'r-.','linewidth',2);
%colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'Z500 anomaly (color)'});
tightmap;
%}

%{
subplot(3,3,4); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(Z850_jja([1:end,1],:,iii))',[50:50:200],'r-','linewidth',2); axis equal tight;
contourm(latt42,lonplot,double(Z850_jja([1:end,1],:,iii))',[-100:50:-50],'r-.','linewidth',2);
%colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'Z850 anomaly (color)'});
tightmap;

subplot(3,3,2); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(T850_jja([1:end,1],:,iii))',[4:2:10],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); caxis([4 10]); %colorbar;
title({'T850 anomaly (color)'});
tightmap;

subplot(3,3,5); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850tend_jja([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850tend (color)'});
tightmap;

subplot(3,3,3); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850AdvU_jja([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850AdvU (color)'});
tightmap;

subplot(3,3,6); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850AdvV_jja([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850AdvV (color)'});
tightmap;

subplot(3,3,9); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850AdvW_jja([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850AdvW (color)'});
tightmap;

subplot(3,3,8); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850err_jja([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850err (color)'});
tightmap;
%}

%
subplot(2,2,4); hold off;
%axesm('MapProjection','pcarree','MapLatLimit',y(nCase)+[-25 25],'MapLonLimit',x(nCase)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
axesm('MapProjection','eqaazim','origin',[y(nCase) x(nCase)],'FLatLimit',[-Inf 50],'grid','on','mlinelocation',20,'plinelocation',20);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
%contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(R_jja([1:end,1],:,iii))',[0,0],'m-','linewidth',2);
contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[1:0.5:1.5]*[sdJJA_NH],'linewidth',2);
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-2:0.5:-1]*[sdJJA_NH],'linewidth',2);
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-100:50:-50],'r-.','linewidth',2);
colormap(gca,jet); caxis([-2 2]*sdJJA_NH); %colorbar;
%colormap(gca,jet(4)); caxis([150 180]); %colorbar;
%title({'\Psi500 anomaly (color)'});
title({'\Psi500 anomaly (1, 1.5\sigma), reversal (magenta)'});
tightmap;
%

%{
subplot(3,3,2); hold off;
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',lont42(round(tmp(3)))+[-90 90]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
contourm(latt42,lonplot,double(PER0202_jja([1:end,1],:,iii))',[0.5,0.5],'r-'); axis equal tight;
hold on;
plotm(coastlat,coastlon,'color',[0.5 0.5 0.5])
%colorbar; colormap(gca,jet(10)); %caxis([-175 325]);
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-');
title({'PER0602 (color)'});
tightmap;

subplot(3,3,8); hold off;
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',lont42(round(tmp(3)))+[-90 90]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
%contourm(latt42,lonplot,double(Wgt7050_jja([1:end,1],:,iii))','r-','showtext','on'); axis equal tight;
contourm(latt42,lonplot,double(Wgt7050_jja([1:end,1],:,iii))',[2:2:8]); axis equal tight;
%contourm(latt42,lonplot,double(Fwrk_jja([1:end,1],:,iii))',[0:20:200],'k-','showtext','on'); axis equal tight;
hold on;
%contourm(latt42,lonplot,double(Wgt7050_jja([1:end,1],:,iii))',[-100:50:-50],'r--','showtext','on');
%contourm(latt42,lonplot,double(Fwrk_jja([1:end,1],:,iii))',[-100:20:-20],'k--','showtext','on');
plotm(coastlat,coastlon,'color',[0.5 0.5 0.5])
colormap(gca,jet(4)); caxis([1 9]); %colorbar;
%colorbar; colormap(gca,jet(10)); %caxis([-175 325]);

contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-');
title({'V850 (color)'});
tightmap;
%}

%title({strTitle, 'Z850 anomaly, T850 anomaly (color)'})

%if(iii==775)
% pause(20);
%disp('pause'); pause;
%end
%disp(iii)
%pause(2)

print(gcf, '-dpsc2','-append',fn_figure);

  end  % loop iii

%% time shape
%{
Hot_cp = reshape(Hotid_jja==HotAttr.id(csort(m)), ds(1),ds(2),[],nyr );
%Hot_cp = reshape(Hot_jja, ds(1),ds(2),[],nyr );
Hot_cp(:,:,[1:6, end-5:end],:) = false;  % used in place of hotT ==T850
Hot_cp(:,latt42(:)<=0,:,:) = false;
ind = find(Hot_cp(:));

HotTs_t = zeros(13,1,'single');
HotT_t = zeros(13,1,'single');
HotW_t = zeros(13,1,'single');
HotDT_t = zeros(13,1,'single');
HotAdvU_t = zeros(13,1,'single');
HotAdvV_t = zeros(13,1,'single');
HotAdvW_t = zeros(13,1,'single');
Hoterr_t = zeros(13,1,'single');
for it = -6:6;
    HotTs_t(it+7) = mean(mx2t_jja(ind + ds(1)*ds(2)*it));
    HotT_t(it+7) = mean(T850_jja(ind + ds(1)*ds(2)*it));
    HotW_t(it+7) = mean(W850_jja(ind + ds(1)*ds(2)*it));
    HotDT_t(it+7) = mean(T850z_jja(ind + ds(1)*ds(2)*it));
    HotAdvU_t(it+7) = mean(T850AdvU_jja(ind + ds(1)*ds(2)*it));
    HotAdvV_t(it+7) = mean(T850AdvV_jja(ind + ds(1)*ds(2)*it));
    HotAdvW_t(it+7) = mean(T850AdvW_jja(ind + ds(1)*ds(2)*it));
    Hoterr_t(it+7) = mean(T850err_jja(ind + ds(1)*ds(2)*it));
end

%figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
clf;
subplot(2,2,1)
plot( -6:6, HotT_t,'-o')
%hold on
%plot( -5.5:5.5, HotT_t(2:end)-HotT_t(1:end-1))
xlabel('time (day)');ylabel('T850 Anomaly (K)');grid on;
title('composite w.r.t. extreme hot event')

subplot(2,2,2)
plot( -6:6, HotW_t,'-o')
xlabel('time (day)');ylabel('\omega850 anomaly (Pa/s)');grid on;
%xlabel('time (day)');ylabel('poleward V850 Anomaly (m/s)');grid on;

subplot(2,2,3)
plot( -6:6, HotDT_t,'-o')
xlabel('time (day)');ylabel('Static stability (K/Pa)');grid on;
%xlabel('time (day)');ylabel('\partial T/\partial y (K/m)');grid on;

subplot(2,2,4)
%plot( -6:6,HotAdvU_t*86400,'c-o', -6:6,HotAdvV_t*86400,'b-v', -6:6,HotAdvW_t*86400,'r-o', -6:6,(HotAdvU_t+HotAdvV_t+HotAdvW_t)*86400,'k-o', -6:6,(Hoterr_t)*86400,'g-o')
%legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal 3d Adv','anomal Error');
plot( -6:6,HotAdvU_t*86400,'c-o', -6:6,HotAdvV_t*86400,'b-v', -6:6,HotAdvW_t*86400,'r-o', -6:6,Hoterr_t*86400,'g-o', -6:6,(HotAdvU_t+HotAdvV_t+HotAdvW_t+Hoterr_t)*86400,'k-o')
legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal Residue','anomal total');
xlabel('time (day)');ylabel('T850 tendency (K/day)');grid on;

%savefig(gcf,['HotTimeEvolution_',ver,'.fig'])
%print(gcf,'-dpdf',['HotTimeEvolution_',ver,'.pdf'])
print(gcf, '-dpsc2','-append',fn_figure);

%}
%xxx end  % loop csort
%    end
%end

system(['ps2pdf ',fn_figure]);
system(['rm ',fn_figure]);

%}

%% case study DJF
%{
clear;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20151231']; text=caseid;
caseid=['ERA-interim_19790101-20151231'];
verX='x903';
load(['temp_',verX,'_',caseid,'.mat']);
disp('finish load'); toc

fn_z500a  = ['../index_wise/Z500_0axx_',caseid,'.mat'];
mat_z500a = matfile(fn_z500a);
%Z500a = mat_z500a.ZaDaily;  % already in latt42/lat0
Z500 = mat_z500a.Z500Daily;  %already in latt42/lat0
Z500a_jja = mat_z500a.Wgt_jja;
Z500a_djf = mat_z500a.Wgt_djf;

%{
%% load nc, check lat lon
nc_z850 = ['../ERA-interim/z850_00Z_',caseid,'.nc'];
Z850 = ncread(nc_z850,'z')/9.81;
lattmp = ncread(nc_z850,'latitude');
lontmp = ncread(nc_z850,'longitude');
if (max(abs(latt42-lattmp))>0.1)
  Z850 = Z850(:,end:-1:1 ,:);
  lattmp = lattmp(end:-1:1);
end
if (max(abs(latt42-lattmp))>0.1 || max(abs(lont42-lontmp))>0.1)
  disp('error')
end

fn_tend = ['../ncl/daily6h_',caseid,'.nc'];
T850  = ncread(fn_tend,'T850Daily');
W850  = ncread(fn_tend,'W850Daily');
T850z = ncread(fn_tend,'S850Daily');
T850AdvU = ncread(fn_tend,'T850AdvU');
T850AdvV = ncread(fn_tend,'T850AdvV');
T850AdvW = ncread(fn_tend,'T850AdvW');
T850err  = ncread(fn_tend,'T850err');
latncl = ncread(fn_tend,'lat');
lonncl = ncread(fn_tend,'lon');
if (max(abs(latt42-latncl))>0.1)
  T850 = T850(:,end:-1:1 ,:);
  W850 = W850(:,end:-1:1 ,:);
  T850z = T850z(:,end:-1:1 ,:);
  T850AdvU = T850AdvU(:,end:-1:1 ,:);
  T850AdvV = T850AdvV(:,end:-1:1 ,:);
  T850AdvW = T850AdvW(:,end:-1:1 ,:);
  T850err = T850err(:,end:-1:1 ,:);
  latncl = latncl(end:-1:1);
end
if (max(abs(latt42-latncl))>0.1 || max(abs(lont42-lonncl))>0.1)
  disp('error')
end
%}

% PER
ver=['2a53_',caseid];
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
load(fn_load2,'PERid_djf','PERdjfAttr');
load(fn_load2,'sdDJF_NH');  %2a53

%{
ver=['7050_',caseid];  %
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
 %load(fn_load2,'timeNan')
PER7050 = (mat_load2.PER);
Wgt7050 = mat_load2.T850Adv3d;

ver=['2631_',caseid]; u=2;v=1;  %Hybrid2020, zprime02, bugfix+4
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
PESR2631 = (squeeze(mat_load2.PESR(:,:,:,u)));
E2631 = (squeeze(mat_load2.E(:,:,:,u)));
%}

%% nan, sign, angle, demean, movmean
%{
%T850 = movmean(T850, prm.D,3);  %TODO necessary??
%timeNan = unique([timeNan, 1:(prm.D-1)/2,ds(3)-(prm.D-1)/2+1:ds(3)]);

% 15/15day*11yr
%
T850_star = movmean(T850, 15,3);  % no Endpoints treatment for star..
T850_star = movmean(T850_star, 15,3);  % no Endpoints treatment for star..
W850_star = movmean(W850, 15,3);  % no Endpoints treatment for star..
W850_star = movmean(W850_star, 15,3);  % no Endpoints treatment for star..
T850AdvU_star = movmean(T850AdvU, 15,3);  % no Endpoints treatment for star..
T850AdvU_star = movmean(T850AdvU_star, 15,3);  % no Endpoints treatment for star..
T850AdvV_star = movmean(T850AdvV, 15,3);  % no Endpoints treatment for star..
T850AdvV_star = movmean(T850AdvV_star, 15,3);  % no Endpoints treatment for star..
T850AdvW_star = movmean(T850AdvW, 15,3);  % no Endpoints treatment for star..
T850AdvW_star = movmean(T850AdvW_star, 15,3);  % no Endpoints treatment for star..
T850err_star = movmean(T850err, 15,3);  % no Endpoints treatment for star..
T850err_star = movmean(T850err_star, 15,3);  % no Endpoints treatment for star..

T850_hat = nan([ds(1:2) 366 nyr-1],'single');
W850_hat = nan([ds(1:2) 366 nyr-1],'single');
T850AdvU_hat = nan([ds(1:2) 366 nyr-1],'single');
T850AdvV_hat = nan([ds(1:2) 366 nyr-1],'single');
T850AdvW_hat = nan([ds(1:2) 366 nyr-1],'single');
T850err_hat = nan([ds(1:2) 366 nyr-1],'single');

for t = 1:366
  tArr = days( datetime('0000-01-01')+caldays(t-1)+calyears(yStart:yEnd) - f_h2d(time(1)) )+1; % order of addition matters
  T850_hat(:,:,t,:) = T850_star(:,:,tArr);
  W850_hat(:,:,t,:) = W850_star(:,:,tArr);
  T850AdvU_hat(:,:,t,:) = T850AdvU_star(:,:,tArr);
  T850AdvV_hat(:,:,t,:) = T850AdvV_star(:,:,tArr);
  T850AdvW_hat(:,:,t,:) = T850AdvW_star(:,:,tArr);
  T850err_hat(:,:,t,:) = T850err_star(:,:,tArr);
end
%T850_hat(:,:,[1:104,end-75:end]) = nan;  % jump elsewhere not Jan 1..
T850_hat = movmean(T850_hat,11,4,'omitnan');
W850_hat = movmean(W850_hat,11,4,'omitnan');
T850AdvU_hat = movmean(T850AdvU_hat,11,4,'omitnan');
T850AdvV_hat = movmean(T850AdvV_hat,11,4,'omitnan');
T850AdvW_hat = movmean(T850AdvW_hat,11,4,'omitnan');
T850err_hat = movmean(T850err_hat,11,4,'omitnan');

dtArr = f_h2d(time); DArr = 366*(dtArr.Year-yStart);
dtArr.Year=0; DArr = DArr + days(dtArr - datetime('0000-01-01') )+1;
%T850 = T850 - T850_bar - T850_hat(:,:,DArr);  % prime
T850 = T850 - T850_hat(:,:,DArr);  % prime
W850 = W850 - W850_hat(:,:,DArr);  % prime
T850AdvU = T850AdvU - T850AdvU_hat(:,:,DArr);  % prime
T850AdvV = T850AdvV - T850AdvV_hat(:,:,DArr);  % prime
T850AdvW = T850AdvW - T850AdvW_hat(:,:,DArr);  % prime
T850err = T850err - T850err_hat(:,:,DArr);  % prime
%T850Cli = T850_hat(:,:,DArr);%T850_bar + 
clear T850_bar T850_star T850_hat tArr dtArr DArr
clear W850_bar W850_star W850_hat
clear T850AdvU_bar T850AdvU_star T850AdvU_hat
clear T850AdvV_bar T850AdvV_star T850AdvV_hat
clear T850AdvW_bar T850AdvW_star T850AdvW_hat
clear T850err_bar T850err_star T850err_hat
%}

%% collect DJF
timeNan = [];

Z500_djf = zeros(ds_djf,'single');
%Z850_djf = zeros(ds_djf,'single');
%T850_djf = zeros(ds_djf,'single');
%W850_djf = zeros(ds_djf,'single');
%T850z_djf = zeros(ds_djf,'single');
%T850AdvU_djf = zeros(ds_djf,'single');
%T850AdvV_djf = zeros(ds_djf,'single');
%T850AdvW_djf = zeros(ds_djf,'single');
%T850err_djf = zeros(ds_djf,'single');
%PER7050_djf = false(ds_djf);
%Wgt7050_djf = zeros(ds_djf,'single');
%PESR2631_djf = zeros(ds_djf,'single');
%E2631_djf = zeros(ds_djf,'single');
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  Z500_djf(:,:,tpointer+(0:tend-tstart)) = Z500(:,:,tstart:tend);
%  Z850_djf(:,:,tpointer+(0:tend-tstart)) = Z850(:,:,tstart:tend);
%  T850_djf(:,:,tpointer+(0:tend-tstart)) = T850(:,:,tstart:tend);
%  W850_djf(:,:,tpointer+(0:tend-tstart)) = W850(:,:,tstart:tend);
%  T850z_djf(:,:,tpointer+(0:tend-tstart)) = T850z(:,:,tstart:tend);
%  T850AdvU_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvU(:,:,tstart:tend);
%  T850AdvV_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvV(:,:,tstart:tend);
%  T850AdvW_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvW(:,:,tstart:tend);
%  T850err_djf(:,:,tpointer+(0:tend-tstart)) = T850err(:,:,tstart:tend);
%  PER7050_djf(:,:,tpointer+(0:tend-tstart)) = PER7050(:,:,tstart:tend);
%  Wgt7050_djf(:,:,tpointer+(0:tend-tstart)) = Wgt7050(:,:,tstart:tend);
%  PESR2631_djf(:,:,tpointer+(0:tend-tstart)) = PESR2631(:,:,tstart:tend);
%  E2631_djf(:,:,tpointer+(0:tend-tstart)) = E2631(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
clear PER7050 Wgt7050 Z500 Z850 PESR2631 E2631  T850 W850 T850z T850AdvU T850AdvV T850AdvW T850err

%TODO some smoothing
R_djf = zeros(ds_djf,'single');
R_djf(:,[yN1:yN2],:) = Z500_djf(:,[yN1:yN2],:) - Z500_djf(:,[yN1:yN2]-5,:);

%% remove trend
%{
  wrk  = nanmean( double(Z850_djf),3); % double precision needed?
  Z850_djf = Z850_djf - repmat( wrk, [1 1 ds_djf(3)]);

% 90day*5yr
%{
T850djf_xyn = squeeze(mean( reshape(T850_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%W850djf_xyn = squeeze(mean( reshape(W850_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850AdvUdjf_xyn = squeeze(mean( reshape(T850AdvU_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850AdvVdjf_xyn = squeeze(mean( reshape(T850AdvV_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850AdvWdjf_xyn = squeeze(mean( reshape(T850AdvW_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
T850errdjf_xyn = squeeze(mean( reshape(T850err_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr

T850_djf = T850_djf - reshape(repmat(reshape(movmean(T850djf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
%W850_djf = W850_djf - reshape(repmat(reshape(movmean(W850djf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850AdvU_djf = T850AdvU_djf - reshape(repmat(reshape(movmean(T850AdvUdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850AdvV_djf = T850AdvV_djf - reshape(repmat(reshape(movmean(T850AdvVdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850AdvW_djf = T850AdvW_djf - reshape(repmat(reshape(movmean(T850AdvWdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
T850err_djf = T850err_djf - reshape(repmat(reshape(movmean(T850errdjf_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
%}

% no quantile
      
% no land

T850tend_djf = T850AdvU_djf+T850AdvV_djf+T850AdvW_djf+T850err_djf;
%}

%{
ind = find( time_djf == hours(datetime(2010,7,28,0,0,0) -datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S')) ) +(-4:5);
%ind = 3359 +(-1:1);

F_y      = repmat(reshape(1:floor(ds(2)/2), [floor(ds(2)/2) 1]), [1 ds(1)]);
%F_y      = repmat(reshape(1:ds(2), [ds(2) 1]), [1 ds(1)]);
%F_y_yxh  = zeros([floor(ds(2)/2) ds(1) 2],'single');
%F_y_yxh(:,:,1) = F_y(1:floor(ds(2)/2)           ,:);
%F_y_yxh(:,:,2) = F_y(end:-1:end-floor(ds(2)/2)+1,:);

F_repmat = repmat(reshape(1:ds(1), [ds(1) 1]), [1 ds(2)]);
F_sin    = sin( F_repmat*2*pi/ds(1)) .*repmat(areaEarth(:)', [ds(1) 1]);
F_cos    = cos( F_repmat*2*pi/ds(1)) .*repmat(areaEarth(:)', [ds(1) 1]);

nBlock = 0; xyBlock =[];
TBlock=0;
S_sin =0; S_cos =0;

for iii = ind(:)'  % row vector
    t = iii;
%    [mh t] = ind2sub(size(bArea_ht), iii);
    if (any(any(Cold_djf(:,:,iii) & repmat(areaEarth(:)'>0, [ds(1) 1]) )))

%     idUniq = unique(Cold_djf(:,:,iii));
%     for id = idUniq(idUniq>0).'
%       % for cyclic boundary condition, find xRef (labels dont cross) to be boundary
%       xRef = find( all(F_id(:,:,t)~=id, 2), 1);
%       if (isempty(xRef))
%           disp(['t=',num2str(t), ', id=',num2str(id)])
%           continue
%       end
%       F_x = xRef + mod(F_repmat-xRef, ds(1));
%       tmp = [t; id; mean(F_x(F_id(:,:,t)==id)); mean(F_y(F_id(:,:,t)==id))];
%         % un-weighted average of x,y index

        S_sin = S_sin +sum(F_sin(Cold_djf(:,:,iii)));
        S_cos = S_cos +sum(F_cos(Cold_djf(:,:,iii)));
        tmp(3) = atan2( sum(F_sin(Cold_djf(:,:,iii))),sum(F_cos(Cold_djf(:,:,iii))) )/2/pi*ds(1);
        tmp = [t; 0; tmp(3); (0)];
%        tmp = [t; 0; tmp(3); ( (ds(2)+1)*(mh==2) +(-1)^(mh==2) *mean(F_y(Cold_djf(:,:,iii))) )];
%        tmp = [t; 0; tmp(3); mean(F_y_yxh( Cold_djf(:,:,iii) ,mh))];
          % un-weighted average of x,y index
        tmp(3) = 0.5 + mod(tmp(3)-0.5, ds(1));  % range from 0.5 - ds(1).5
        xyBlock = [xyBlock tmp];
    end
end

        tmp(3) = atan2( S_sin,S_cos )/2/pi*ds(1);
        tmp(3) = 0.5 + mod(tmp(3)-0.5, ds(1));  % range from 0.5 - ds(1).5

  tmp(3) = 40/360*ds(1);
%}

%Coldid_djf(Coldid_djf==0) = nan;
addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added
%rng(2017);

fn_figure = ['case.ps'];
%fn_figure = ['case',datestr( f_h2d(mean(time_djf(ind))),'yyyymmdd'),'.ps'];
system(['rm ',fn_figure]);

% area time series
%{
figure('units','inches','position',[0 1 12 9], 'paperUnits','inches','papersize',[12 9],'paperposition',[0 0 12 9]);

%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];

%wrk = (hours(mean(time_djf(ind)))+datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
%yyyy = wrk.Year;
yyyy = ceil(mean(ind)/nd_djf);  % index starting 1
wrk = (yyyy-1)*nd_djf +(1:nd_djf);
%yyyy=2014;
%wrk=(yyyy-yStart)*nd_djf+(1:nd_djf);
%figure; plot(wrk,bdjfArea_t(wrk),'k-', wrk,coldArea_t(wrk),'r-');

%plot(mod(ind,nd_djf),bdjfArea_t(ind));

ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,1); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1); ax.ColorOrderIndex=yyyy;
%xlabel({'Blocked area on hemisphere (km^2)'}); ylabel('Extreme area on hemisphere (km^2)');
ylabel('Extreme area (km^2)'); %xlabel({'day'});
%plot(reshape( coldArea_t,[],nyr-1 ));
%plot([1:nd_djf],coldArea_t(wrk));
plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
%plot([1 nd_djf],ColdQuantile(i,j)*[1 1],'k-');
title([num2str(yyyy+yStart-1)]);

ver=['0602_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,5); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bdjfArea_t,[],nyr-1 ));
%plot([1:nd_djf],bdjfArea_t(wrk));
plot(datenum(f_h2d(time_djf(wrk))),bdjfArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('Dunn-Sigouin and Son 2013');

ver=['2631_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,6); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bdjfArea_t,[],nyr-1 ));
%plot([1:nd_djf],bdjfArea_t(wrk));
plot(datenum(f_h2d(time_djf(wrk))),bdjfArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('Hassanzadeh et al. 2014');

ver=['4001_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,9); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bdjfArea_t,[],nyr-1 ));
%plot([1:nd_djf],bdjfArea_t(wrk));
plot(datenum(f_h2d(time_djf(wrk))),bdjfArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('Scherrer et al. 2006');

ver=['7050_',caseid];
  load(['scatter_',textTH,'_',ver,'.mat'])
subplot(3,4,10); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1); ax.ColorOrderIndex=yyyy;
ylabel('Blocked area (km^2)'); %xlabel({'day'});
%plot(reshape( bdjfArea_t,[],nyr-1 ));
%plot([1:nd_djf],bdjfArea_t(wrk));
plot(datenum(f_h2d(time_djf(wrk))),bdjfArea_t(wrk)); datetick('x','mm/dd'); grid on;
title('v850 quantile (7050)');

textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
print(gcf, '-dpsc2','-append',fn_figure);
%}

%% map
%
disp('start map'); toc
figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
%figure('Position',[0 80 1920 1000],'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%figure('OuterPosition',[0 80 1366 700]);
%'meridianlabel','on','parallellabel','on','mlinelocation',60,'plinelocation',30,

%strcmp(ColdAttr.id,'2308357')
%m=1; csort=594;
Multid_djf = removecats(Coldid_djf.*circshift(PERid_djf,[11 -5 0]));  %TODO
ids = str2double(split(categories(Multid_djf)));
numPER_Cold = countcats(categorical(ids(:,1),str2double(ColdAttr.id)));
areaCold_PER = zeros([numel(PERdjfAttr.id) 1],'single');
for m = 1:size(ids,1)
  areaCold_PER(ids(m,2)) = areaCold_PER(ids(m,2))+ColdAttr.area(ids(m,1))/numPER_Cold(ids(m,1));
end

%csort = find(numPER_Cold>1);  %split
%[~,csort] = sort(ColdAttr.area,'descend'); csort=csort(1:11);  % csort
%[~,csort] = sort(ColdAttr.area.*(~numPER_Cold),'descend'); csort=csort(1:10);  % missed event
numCold_PER = countcats(categorical(ids(:,2),str2double(PERdjfAttr.id)));
[~,csort] = sort(PERdjfAttr.arealsm.*(~numCold_PER),'descend'); csort=csort(1:10);  %False alarm
%csort = [169;147]; %False alarm
%[~,csort] = sort(areaCold_PER,'descend'); csort=csort(1:5);  %Strong case
%csort = [171;89]; %Good Catch
%[~,csort] = sort(PERdjfAttr.arealsm.*(ceil(PERdjfAttr.tstart/nd_djf)==(1991-yStart+1)),'descend'); csort=csort(1:6);  %1991
%[~,csort] = sort(ColdAttr.area.*(ceil(ColdAttr.tstart/nd_djf)==(1991-yStart+1)),'descend'); csort=csort(1:10);  %1991
%disp('pause'); pause;
%csort = [445;453;454]; %1991

%tstart = PERdjfAttr.tstart(csort); tend = PERdjfAttr.tend(csort);
%x = lont42(round(ColdAttr.x(csort)))-15; y = latt42(round(ColdAttr.y(csort)))+7.5;
x = lont42(round(PERdjfAttr.x(csort)))+15; y = latt42(round(PERdjfAttr.y(csort)))-7.5;
 % :%s/latt42(round(PERdjfAttr\.y(csort(m))))/y(m)/gce | %s/lont42(round(PERdjfAttr\.x(csort(m))))/x(m)/gce
%ind = find( time_djf == hours(datetime(1995,7,12,0,0,0) -datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S')) ) +(-1:5); y=42; x=-88; csort=csort(2);

%for iii = ind(:)'  % row vector
 %   [mh t] = ind2sub(size(bArea_ht), iii);
 %   if (any(any(Cold_djf(:,:,iii) & repmat(areaEarth(:)'>0, [ds(1) 1]) )))
for m = 1:numel(csort)
 %  for iii = PERdjfAttr.tstart(csort(m))-2:PERdjfAttr.tend(csort(m))+2
 %for iii = 1:nd_djf
%tstart = min([ColdAttr.tstart(csort(m));PERdjfAttr.tstart( ids(ids(:,1)==csort(m),2) )]);
%tend   = max([ColdAttr.tend(csort(m));  PERdjfAttr.tend( ids(ids(:,1)==csort(m),2) )]);
tstart = min([PERdjfAttr.tstart(csort(m));ColdAttr.tstart( ids(ids(:,2)==csort(m),1) )]);
tend   = max([PERdjfAttr.tend(csort(m));  ColdAttr.tend( ids(ids(:,2)==csort(m),1) )]);
for iii = tstart-1:tend+1
 %PERdjfAttr.tmax(csort(m))
clf;

%pcolor id
%
subplot(2,2,1); hold off;
%axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
axesm('MapProjection','eqaazim','origin',[y(m) x(m)],'FLatLimit',[-Inf 50],'grid','on','mlinelocation',20,'plinelocation',20);
%contourm(latt42,lonplot,double(Coldid_djf([1:end,1],:,iii))',[0.5 6.5]); axis equal tight;
pcolormPC(latt42,lont42,Coldid_djf(:,:,iii)');
hold on;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
colormap(gca,jet); caxis([csort(m)-10 csort(m)+5]); colorbar; caxis auto;

%contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-');
%title({'Coldid (color)'});
title({['Coldid (color) ',datestr( f_h2d(time_djf(iii)) )]});
%title({['Coldid (color) ','iii=',num2str(iii)]});
tightmap;
%

subplot(2,2,3); hold off;
%axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
axesm('MapProjection','eqaazim','origin',[y(m) x(m)],'FLatLimit',[-Inf 50],'grid','on','mlinelocation',20,'plinelocation',20);
%contourm(latt42,lonplot,double(PERid_djf([1:end,1],:,iii))',[0.5 6.5]); axis equal tight;
pcolormPC(latt42,lont42,PERid_djf(:,:,iii)');
hold on;
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
colormap(gca,jet); caxis([csort(m)-5 csort(m)+5]); colorbar; %caxis auto;

%contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-');
title({'PERid (color)'});
%title({['PERid (color) ',datestr( f_h2d(time_djf(iii)) )]});
tightmap;

%title
%{
subplot(3,3,1); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(mn2t_djf([1:end,1],:,iii))',[4:2:10],'linewidth',2); axis equal tight;
colormap(gca,jet); caxis([4 10]); %colorbar;
title({['mn2t (color) ','iii=',num2str(iii)]});
%title({['mn2t (color) ',datestr( f_h2d(time_djf(iii)) )]});
tightmap;
%}

%template
%{
subplot(3,3,4); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(Z500_djf([1:end,1],:,iii))',[5000:100:5800],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'Z500 (color)'});
tightmap;

subplot(3,3,7); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[50:50:200],'r-','linewidth',2); axis equal tight;
contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-100:50:-50],'r-.','linewidth',2);
%colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'Z500 anomaly (color)'});
tightmap;
%}

%{
subplot(3,3,4); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(Z850_djf([1:end,1],:,iii))',[50:50:200],'r-','linewidth',2); axis equal tight;
contourm(latt42,lonplot,double(Z850_djf([1:end,1],:,iii))',[-100:50:-50],'r-.','linewidth',2);
%colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'Z850 anomaly (color)'});
tightmap;

subplot(3,3,2); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(T850_djf([1:end,1],:,iii))',[4:2:10],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); caxis([4 10]); %colorbar;
title({'T850 anomaly (color)'});
tightmap;

subplot(3,3,5); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850tend_djf([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850tend (color)'});
tightmap;

subplot(3,3,3); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850AdvU_djf([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850AdvU (color)'});
tightmap;

subplot(3,3,6); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850AdvV_djf([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850AdvV (color)'});
tightmap;

subplot(3,3,9); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850AdvW_djf([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850AdvW (color)'});
tightmap;

subplot(3,3,8); hold off;
axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850err_djf([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850err (color)'});
tightmap;
%}

subplot(2,2,2); hold off;
%axesm('MapProjection','pcarree','MapLatLimit',y(m)+[-25 25],'MapLonLimit',x(m)+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
axesm('MapProjection','eqaazim','origin',[y(m) x(m)],'FLatLimit',[-Inf 50],'grid','on','mlinelocation',20,'plinelocation',20);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
%contourm(latt42,lonplot,double(R_djf([1:end,1],:,iii))',[0,0],'m-','linewidth',2);
contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[1:0.5:2]*[sdDJF_NH],'linewidth',2);
contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-2:0.5:-1]*[sdDJF_NH],'linewidth',2);
%contourm(latt42,lonplot,double(Z500a_djf([1:end,1],:,iii))',[-100:50:-50],'r-.','linewidth',2);
colormap(gca,jet); caxis([-2 2]*sdDJF_NH); %colorbar;
%colormap(gca,jet(4)); caxis([150 180]); %colorbar;
title({'\Psi500 anomaly (color)'});
tightmap;

%{
subplot(3,3,2); hold off;
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',lont42(round(tmp(3)))+[-90 90]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
contourm(latt42,lonplot,double(PER0202_djf([1:end,1],:,iii))',[0.5,0.5],'r-'); axis equal tight;
hold on;
plotm(coastlat,coastlon,'color',[0.5 0.5 0.5])
%colorbar; colormap(gca,jet(10)); %caxis([-175 325]);
contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-');
title({'PER0602 (color)'});
tightmap;

subplot(3,3,8); hold off;
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',lont42(round(tmp(3)))+[-90 90]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
%contourm(latt42,lonplot,double(Wgt7050_djf([1:end,1],:,iii))','r-','showtext','on'); axis equal tight;
contourm(latt42,lonplot,double(Wgt7050_djf([1:end,1],:,iii))',[2:2:8]); axis equal tight;
%contourm(latt42,lonplot,double(Fwrk_djf([1:end,1],:,iii))',[0:20:200],'k-','showtext','on'); axis equal tight;
hold on;
%contourm(latt42,lonplot,double(Wgt7050_djf([1:end,1],:,iii))',[-100:50:-50],'r--','showtext','on');
%contourm(latt42,lonplot,double(Fwrk_djf([1:end,1],:,iii))',[-100:20:-20],'k--','showtext','on');
plotm(coastlat,coastlon,'color',[0.5 0.5 0.5])
colormap(gca,jet(4)); caxis([1 9]); %colorbar;
%colorbar; colormap(gca,jet(10)); %caxis([-175 325]);

contourm(latt42,lonplot,double(Cold_djf([1:end,1],:,iii))',[0.5,0.5],'k-');
title({'V850 (color)'});
tightmap;
%}

%title({strTitle, 'Z850 anomaly, T850 anomaly (color)'})

%if(iii==775)
% pause(20);
%disp('pause'); pause;
%end
%disp(iii)
%pause(2)

print(gcf, '-dpsc2','-append',fn_figure);

  end  % loop iii

%% time shape
%{
Cold_cp = reshape(Coldid_djf==ColdAttr.id(csort(m)), ds(1),ds(2),[],nyr-1 );
%Cold_cp = reshape(Cold_djf, ds(1),ds(2),[],nyr-1 );
Cold_cp(:,:,[1:6, end-5:end],:) = false;  % used in place of coldT ==T850
Cold_cp(:,latt42(:)<=0,:,:) = false;
ind = find(Cold_cp(:));

ColdTs_t = zeros(13,1,'single');
ColdT_t = zeros(13,1,'single');
ColdW_t = zeros(13,1,'single');
ColdDT_t = zeros(13,1,'single');
ColdAdvU_t = zeros(13,1,'single');
ColdAdvV_t = zeros(13,1,'single');
ColdAdvW_t = zeros(13,1,'single');
Colderr_t = zeros(13,1,'single');
for it = -6:6;
    ColdTs_t(it+7) = mean(mn2t_djf(ind + ds(1)*ds(2)*it));
    ColdT_t(it+7) = mean(T850_djf(ind + ds(1)*ds(2)*it));
    ColdW_t(it+7) = mean(W850_djf(ind + ds(1)*ds(2)*it));
    ColdDT_t(it+7) = mean(T850z_djf(ind + ds(1)*ds(2)*it));
    ColdAdvU_t(it+7) = mean(T850AdvU_djf(ind + ds(1)*ds(2)*it));
    ColdAdvV_t(it+7) = mean(T850AdvV_djf(ind + ds(1)*ds(2)*it));
    ColdAdvW_t(it+7) = mean(T850AdvW_djf(ind + ds(1)*ds(2)*it));
    Colderr_t(it+7) = mean(T850err_djf(ind + ds(1)*ds(2)*it));
end

%figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%figure('units','inches','position',[0 1 16 9], 'paperUnits','inches','papersize',[16 9],'paperposition',[0 0 16 9]);
clf;
subplot(2,2,1)
plot( -6:6, ColdT_t,'-o')
%hold on
%plot( -5.5:5.5, ColdT_t(2:end)-ColdT_t(1:end-1))
xlabel('time (day)');ylabel('T850 Anomaly (K)');grid on;
title('composite w.r.t. extreme cold event')

subplot(2,2,2)
plot( -6:6, ColdW_t,'-o')
xlabel('time (day)');ylabel('\omega850 anomaly (Pa/s)');grid on;
%xlabel('time (day)');ylabel('poleward V850 Anomaly (m/s)');grid on;

subplot(2,2,3)
plot( -6:6, ColdDT_t,'-o')
xlabel('time (day)');ylabel('Static stability (K/Pa)');grid on;
%xlabel('time (day)');ylabel('\partial T/\partial y (K/m)');grid on;

subplot(2,2,4)
%plot( -6:6,ColdAdvU_t*86400,'c-o', -6:6,ColdAdvV_t*86400,'b-v', -6:6,ColdAdvW_t*86400,'r-o', -6:6,(ColdAdvU_t+ColdAdvV_t+ColdAdvW_t)*86400,'k-o', -6:6,(Colderr_t)*86400,'g-o')
%legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal 3d Adv','anomal Error');
plot( -6:6,ColdAdvU_t*86400,'c-o', -6:6,ColdAdvV_t*86400,'b-v', -6:6,ColdAdvW_t*86400,'r-o', -6:6,Colderr_t*86400,'g-o', -6:6,(ColdAdvU_t+ColdAdvV_t+ColdAdvW_t+Colderr_t)*86400,'k-o')
legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal Residue','anomal total');
xlabel('time (day)');ylabel('T850 tendency (K/day)');grid on;

%savefig(gcf,['ColdTimeEvolution_',ver,'.fig'])
%print(gcf,'-dpdf',['ColdTimeEvolution_',ver,'.pdf'])
print(gcf, '-dpsc2','-append',fn_figure);

%}
end  % loop csort
%    end
%end

system(['ps2pdf ',fn_figure]);
system(['rm ',fn_figure]);

%}

%:'<,'>s/jja\C/djf/gce | '<,'>s/JJA\C/DJF/gce | '<,'>s/Hot\C/Cold/gce | '<,'>s/hot\C/cold/gce | '<,'>s/mx2t/mn2t/gce | '<,'>s/yStart:yEnd/yStart+1:yEnd/gce | '<,'>s/yyyy-yStart+1/yyyy-yStart/gce | '<,'>s?/nyr?/(nyr)?gce | '<,'>s/nyr/&-1/gce | noh

%:'<,'>s/nccreate(fn_savenc,'\([^']*\)',.*$/ncwrite(fn_savenc,'\1',\1)/ | noh

% : set fdm=expr foldexpr=getline(v\:lnum)=~'^%%.*$'?0\:1:
% vim: set fdm=marker foldmarker=%{,%}:

