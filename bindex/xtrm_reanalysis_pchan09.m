%% define extremes, for F32 grid
% TODO 09.m, region tune, hottest 60 days, label contiguous, 15/15day*11yr, mimic BlockStat*.m
% matlab -nosplash -nodesktop -r "text='abs1.0_T85_CC1';thresh={0.01,'quantile',5}; xtrm_colocate_pchan; exit"
% 'module load matlab; \matlab -nojvm -nodesktop -nodisplay -nosplash -r "xtrm_reanalysis_pchan09;exit" >& slurm-${SLURM_JOB_NAME}-${SLURM_JOBID}' --mail-type=END
% sbatch --account=kuang_lab -p huce_intel -J x919 -n 1 -t 1200 --mem=30000 -o "slurm" --wrap='\matlab -nodesktop -nodisplay -nosplash -r "xtrm_reanalysis_pchan09;exit" >& slurm-${SLURM_JOB_NAME}-${SLURM_JOBID}' --mail-type=END

%% load and save data
%
tic
setenv('LD_LIBRARY_PATH',getenv('LD_LIBRARY_save'));
%cd ../matlab

%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%thresh={0.01,'quantile',5,5,30}; caseid=['ERA-interim_19790101-20171231'];
%thresh={0.01,'quantile',5,5,40}; caseid=['ERA-interim_19790101-20171231'];  %x903
%thresh={0.88,'quantile',5,5,40}; caseid=['ERA-interim_19790101-20171231'];  %x904
%thresh={1.5,'sigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x905
%thresh={2,'sigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x906
%thresh={2.5,'sigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x907
%thresh={3,'sigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x908
%thresh={0.01,'quantile',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x909
%thresh={0.01,'quantile',5,0,45}; caseid=['ERA-interim_19790101-20171231'];  %x910
%thresh={0.88,'quantile',5,3.0199504696681*3,40}; caseid=['ERA-interim_19790101-20171231'];  %x911
%thresh={0.88,'quantile',5,5,40}; caseid=['ERA-interim_19790101-20171231'];  %x912
%thresh={0.88,'quantile',5,5,40}; caseid=['ERA-interim_19790101-20171231'];  %x913
%thresh={0.88,'quantile',5,9,40}; caseid=['ERA-interim_19790101-20171231'];  %x914
%thresh={2,'localsigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x915
%thresh={3,'localsigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x916
%thresh={2.3,'localsigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x917
%thresh={2.3,'localsigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x918
thresh={2.3,'localsigma',5,0,40}; caseid=['ERA-interim_19790101-20171231'];  %x919
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
 % 1: number for quantile/sigma
 % 2: 'quantile' or 'sigma'
 % 3: persistence (d)
 % 4: cutoff by temperature (K)
 % 5: cutoff by latitude

prm.yS1  = 1;  % 87.86deg
prm.yS2  = 32; % xx.xxdeg  thresh{5}
prm.yN1  = 64+1-prm.yS2;
prm.yN2  = 64+1-prm.yS1;

prm.A1 = thresh{1};
prm.Am = thresh{2};
prm.A = thresh{4};
prm.D = thresh{3};

yS1=prm.yS1; yS2=prm.yS2; yN1=prm.yN1; yN2=prm.yN2;

ver=['x919_',caseid];  %
verX=ver(1:4);
%strTitle = 'x900: Extreme, 15/15day*11yr, 1% 5K 0lat';  %, 00Z
%strTitle = 'x901: Extreme, 15/15day*11yr, 0.5% 5K 0lat';  %, 00Z
%strTitle = 'x902: Extreme, 15/15day*11yr, 1% 5K 30lat';  %, 00Z
%strTitle = 'x903: Extreme, 15/15day*11yr, 1% 5K 40lat';  %, 00Z
%strTitle = 'x904: Extreme, 15/15day*11yr, -% 5K 40lat';  %, 00Z
%strTitle = 'x905: Extreme, 15/15day*11yr, 1.5std 0K 40lat';  %, 00Z
%strTitle = 'x906: Extreme, 15/15day*11yr, 2std 0K 40lat';  %, 00Z
%strTitle = 'x907: Extreme, 15/15day*11yr, 2.5std 0K 40lat';  %, 00Z
%strTitle = 'x908: Extreme, 15/15day*11yr, 3std 0K 40lat';  %, 00Z
%strTitle = 'x909: Extreme, 15/15day*11yr, 1% 0K 40lat';  %, 00Z
%strTitle = 'x910: Extreme, 15/15day*11yr, 1% 0K 45lat';  %, 00Z
%strTitle = 'x911: Extreme, 15/15day*11yr, -% 9.06K 40lat';  %, 00Z
%strTitle = 'x912: Extreme, 15/15day*11yr, -% 5K 40lat';  %, 00Z
%strTitle = 'x913: tododododExtreme, 15/15day*11yr, -% 5K 40lat';  %, 00Z
%strTitle = 'x914: Extreme, 15/15day*11yr, -% 9K 40lat';  %, 00Z
%strTitle = 'x915: Extreme, 15/15day*11yr, 2localstd 0K 40lat';  %, 00Z
%strTitle = 'x916: Extreme, 15/15day*11yr, 3localstd 0K 40lat';  %, 00Z
%strTitle = 'x917: Extreme, 15/15day*11yr, 2.3localstd 0K 40lat';  %, 00Z
%strTitle = 'x918: Extreme, 15/15day*11yr, 2.3localstd 0K 50lat';  %, 00Z
strTitle = 'x919: Extreme, 15/15day*11yr, 2.3localstd 0K 40lat';  %, 00Z

mkdir('../index_wise');
fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
%fn_load0 = ['../sks/int_z500_zg_day_',caseid,'N.nc'];  % 161021, 00Z
fn_load0 = ['../ERA-interim/zg_day_',caseid,'.nc'];
nc_load1 = ['../ERA-interim/mx2t_3h_fc9-18_',caseid,'.nc'];
%nc_load1 = @(fc) ['../ERA-interim/mx2t_12h_fc',num2str(fc),'_',caseid,'.nc'];
%nc_load1b = @(fc) ['../ERA-interim/mn2t_12h_fc',num2str(fc),'_',caseid,'.nc'];
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
time = ncread(fn_load0,'time');
mx2t = nan(128,64,nDays,'single');
mn2t = nan(128,64,nDays,'single');
ds = size(mx2t);
%for fc = 9:3:18;
%  wrk = ncread(nc_load1(fc),'mx2t',[1 1 2-(fc>12)],[Inf Inf 2*(ds(3)-1)]);
 %  wrk = ncread(nc_load1(fc),'mx2t',[1 1 3-(fc>12)],[Inf Inf 2*(ds(3)-1)]);  %12Z
%  mx2t(:,:,2:end) = max(mx2t(:,:,2:end), squeeze(max( reshape(wrk,[ds(1:2) 2 ds(3)-1]), [],3)) );
%  wrk = ncread(nc_load1b(fc),'mn2t',[1 1 2-(fc>12)],[Inf Inf 2*(ds(3)-1)]);
 %  wrk = ncread(nc_load1b(fc),'mn2t',[1 1 3-(fc>12)],[Inf Inf 2*(ds(3)-1)]);  %12Z
%  mn2t(:,:,2:end) = min(mn2t(:,:,2:end), squeeze(min( reshape(wrk,[ds(1:2) 2 ds(3)-1]), [],3)) );
%end
wrk = ncread(nc_load1,'mx2t',[1 1 3],[Inf Inf 8*(ds(3)-1)]);
mx2t(:,:,2:end) = squeeze(max( reshape(wrk,[ds(1:2) 8 ds(3)-1]), [],3));
wrk = ncread(nc_load1,'mn2t',[1 1 3],[Inf Inf 8*(ds(3)-1)]);
mn2t(:,:,2:end) = squeeze(min( reshape(wrk,[ds(1:2) 8 ds(3)-1]), [],3));
mx2t(:,:,1) = mx2t(:,:,2);  % a good fillvalue..
mn2t(:,:,1) = mn2t(:,:,2);  % a good fillvalue..
clear wrk

lsm = (ncread(nc_load9,'lsm')==1);  % double->logical, 2d

% load blocking
%{
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
%}
timeNan = [];
disp('finish load'); toc

%% check lat lon, referencing fn_t42
latt42  = ncread(fn_t42,'lat');
lat1  = ncread(nc_load1,'latitude');
%lat1  = ncread(nc_load1(9),'latitude');
%lat1b  = ncread(nc_load1b(9),'latitude');
%lat2  = ncread(fn_load2,'latitude');
%lat3  = ncread(nc_load3,'latitude');
lat9  = ncread(nc_load9,'latitude');

lont42  = ncread(fn_t42,'lon');
lon1  = ncread(nc_load1,'longitude');
%lon1  = ncread(nc_load1(9),'longitude');
%lon1b  = ncread(nc_load1b(9),'longitude');
%lon2  = ncread(fn_load2,'longitude');
%lon3  = ncread(nc_load3,'longitude');
lon9  = ncread(nc_load9,'longitude');

if (lat1(2)<lat1(1))
  mx2t = mx2t(:,end:-1:1 ,:);
  mn2t = mn2t(:,end:-1:1 ,:);
  lat1 = lat1(end:-1:1);
end
%if (lat1b(2)<lat1b(1))
%  mn2t = mn2t(:,end:-1:1 ,:);
%  lat1b = lat1b(end:-1:1);
%end
if (lat9(2)<lat9(1))
  lsm = lsm(:,end:-1:1);
  lat9 = lat9(end:-1:1);
end
%if (any(lont42<0) || max(abs(latt42-lat1))>0.1 || max(abs(lont42-lon1))>0.1 || max(abs(latt42-lat1b))>0.1 || max(abs(lont42-lon1b))>0.1 || max(abs(latt42-lat9))>0.1 || max(abs(lont42-lon9))>0.1)
if (any(lont42<0) || max(abs(latt42-lat1))>0.1 || max(abs(lont42-lon1))>0.1 || max(abs(latt42-lat9))>0.1 || max(abs(lont42-lon9))>0.1)
  disp('error'); return;
end

    lat_bnds = ncread(fn_t42,'lat_bnds');
    R = 6371; %km
%    areaEarth = (2*pi)*(R^2)* (cosd(lat1a') *1*pi/180);
    areaEarth = (2*pi)*(R^2)* (sind(lat_bnds(2,:))-sind(lat_bnds(1,:)));
%    areaEarth = areaEarth(1:floor(ds_jja(2)/2));
%  areaEarth(latt42(:)<=thresh{5})=0;
  areaEarth(latt42(:)<=0)=0;

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

% movmean
mx2t = movmean(mx2t, thresh{3},3);
mn2t = movmean(mn2t, thresh{3},3);
timeNan = unique([timeNan, 1:(thresh{3}-1)/2,ds(3)-(thresh{3}-1)/2+1:ds(3)]);

%save(['wrk_',ver,'.mat'],'-v7.3');  %TODO
%

% can use movmean instead..
%{
mx2t = filter(ones(1,thresh{3}), thresh{3}, mx2t, [],3);
mx2t(:,:, (1+(thresh{3}-1)/2):(end-(thresh{3}-1)/2)) = mx2t(:,:,thresh{3}:end);
mx2t(:,:, [1:(thresh{3}-1)/2,end-(thresh{3}-1)/2+1:end]) = nan;
mn2t = filter(ones(1,thresh{3}), thresh{3}, mn2t, [],3);
mn2t(:,:, (1+(thresh{3}-1)/2):(end-(thresh{3}-1)/2)) = mn2t(:,:,thresh{3}:end);
mn2t(:,:, [1:(thresh{3}-1)/2,end-(thresh{3}-1)/2+1:end]) = nan;
timeNan = unique([timeNan, 1:(thresh{3}-1)/2,ds(3)-(thresh{3}-1)/2+1:ds(3)]);
%}

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
  return;
end

  Hot( abs(mx2t) <thresh{4} ) =false;
  Cold( abs(mn2t) <thresh{4} ) =false;

  Hot = Hot & repmat(lsm, [1 1 ds(3)]);
  Cold = Cold & repmat(lsm, [1 1 ds(3)]);

  lsm_jja=lsm; lsm_djf=lsm;
%}

%
%clear;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%ver=['x919_',caseid];  %
%load(['wrk_',ver,'.mat']);  %TODO
%disp('finish load wrk'); toc

%% find hottest days
%{
lsm_jja = lsm; lsm_jja(:,latt42(:)<=thresh{5})=false;
area_jja = lsm_jja*diag(areaEarth)/ds(1);
wrk=reshape(mx2t,ds(1)*ds(2),ds(3));
mx2t_t = nansum(repmat(area_jja(:),[1 ds(3)]).*wrk,1)./sum(area_jja(:));

mx2t_bar_t = movmean(mx2t_t,365,2,'Endpoints','fill');
mx2t_bar_t(1:182) = mx2t_bar_t(183);
mx2t_bar_t(end-181:end) = mx2t_bar_t(end-182);
mx2t_t = mx2t_t - mx2t_bar_t;
for t = 1:92
  tArr = days( datetime('0000-06-01')+caldays(t-1)+calyears(yStart:yEnd) - f_h2d(time(1)) )+1; % order of addition matters
  mx2t_d(t) = mean(mx2t_t(tArr));
end
%subplot(2,2,1);
plot(mx2t_d,'o-'); grid on;
xlabel({'day'}); title({''});
[~,wrk]=sort(mx2t_d,'descend'); disp(wrk(59:60))  % 20 79
disp('return'); return;
%}


%% SKS95 20170522
%{
mx2t_bar = movmean(mx2t,365,3,'Endpoints','fill');
mx2t_bar(:,:,1:182) = repmat( mx2t_bar(:,:,183), [1 1 182]);
mx2t_bar(:,:,end-181:end) = repmat( mx2t_bar(:,:,end-182), [1 1 182]);
disp('finish sks05'); toc
%mx2t = mx2t - mx2t_bar;
mx2t_star = movmean(mx2t - mx2t_bar, 31,3);  % no Endpoints treatment for star..
mx2t_hat = nan([ds(1:2) 366 nyr],'single');

mn2t_bar = movmean(mn2t,365,3,'Endpoints','fill');
mn2t_bar(:,:,1:182) = repmat( mn2t_bar(:,:,183), [1 1 182]);
mn2t_bar(:,:,end-181:end) = repmat( mn2t_bar(:,:,end-182), [1 1 182]);
%mn2t = mn2t - mn2t_bar;
mn2t_star = movmean(mn2t - mn2t_bar, 31,3);  % no Endpoints treatment for star..
mn2t_hat = nan([ds(1:2) 366 nyr],'single');
%}

%mx2t_star = movmean(mx2t, 31,3);  % no Endpoints treatment for star..
%mn2t_star = movmean(mn2t, 31,3);  % no Endpoints treatment for star..
mx2t_star = movmean(mx2t, 15,3);  % no Endpoints treatment for star..
mx2t_star = movmean(mx2t_star, 15,3);  % no Endpoints treatment for star..
mn2t_star = movmean(mn2t, 15,3);  % no Endpoints treatment for star..
mn2t_star = movmean(mn2t_star, 15,3);  % no Endpoints treatment for star..

mx2t_hat = nan([ds(1:2) 366 nyr],'single');
mn2t_hat = nan([ds(1:2) 366 nyr],'single');

disp('finish sks20'); toc

for t = 1:366
  tArr = days( datetime('0000-01-01')+caldays(t-1)+calyears(yStart:yEnd) - f_h2d(time(1)) )+1; % order of addition matters
  mx2t_hat(:,:,t,:) = mx2t_star(:,:,tArr);
  mn2t_hat(:,:,t,:) = mn2t_star(:,:,tArr);
end
disp('finish sks30'); toc
mn2t_hat(:,:,[1:104,end-75:end]) = nan;  % jump elsewhere not Jan 1..
mx2t_hat = movmean(mx2t_hat,11,4,'omitnan');
mn2t_hat = movmean(mn2t_hat,11,4,'omitnan');
disp('finish sks40'); toc

dtArr = f_h2d(time); DArr = 366*(dtArr.Year-yStart);
dtArr.Year=0; DArr = DArr + days(dtArr - datetime('0000-01-01') )+1;
%mx2t = mx2t - mx2t_bar - mx2t_hat(:,:,DArr);  % prime
%mn2t = mn2t - mn2t_bar - mn2t_hat(:,:,DArr);
mx2t = mx2t - mx2t_hat(:,:,DArr);  % prime
mn2t = mn2t - mn2t_hat(:,:,DArr);
mx2tCli = mx2t_hat(:,:,DArr);%mx2t_bar + 
mn2tCli = mn2t_hat(:,:,DArr);%mn2t_bar + 
%clear mx2t_bar mx2t_star mx2t_hat mn2t_bar mn2t_star mn2t_hat tArr dtArr DArr TODO
disp('finish sks'); toc

%% collect JJA
hJJAstart = hours(datetime(yStart:yEnd,6,20,0,0,0) - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
hJJAend   = hours(datetime(yStart:yEnd,8,18,0,0,0) - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
ds_jja=ds; ds_jja(3) = sum(hJJAend-hJJAstart)/24 +numel(hJJAstart); nd_jja=ds_jja(3)/nyr;

mx2t_jja = zeros(ds_jja,'single');
mx2tCli_jja = zeros(ds_jja,'single');
WgtStar_jja = zeros(ds_jja,'single');
%Z500a_jja = zeros(ds_jja,'single');
%Hot_jja = false(ds_jja);
%PER0202_jja = false(ds_jja);
%PER2225_jja = false(ds_jja);
%PER4001_jja = false(ds_jja);
%pv_jja  = zeros(ds_jja,'single');
%PER5000 = false(ds_jja);
%PER5001 = false(ds_jja);
time_jja = zeros([ds_jja(3) 1], class(time));
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  mx2t_jja(:,:,tpointer+(0:tend-tstart)) = mx2t(:,:,tstart:tend);
  mx2tCli_jja(:,:,tpointer+(0:tend-tstart)) = mx2tCli(:,:,tstart:tend);
  WgtStar_jja(:,:,tpointer+(0:tend-tstart)) = mx2t_star(:,:,tstart:tend);
%  Z500a_jja(:,:,tpointer+(0:tend-tstart)) = Z500a(:,:,tstart:tend);
%  Hot_jja(:,:,tpointer+(0:tend-tstart)) = Hot(:,:,tstart:tend);  % old: all season

%  PER0202_jja(:,:,tpointer+(0:tend-tstart)) = PER0202(:,:,tstart:tend);
%  PER2225_jja(:,:,tpointer+(0:tend-tstart)) = PER2225(:,:,tstart:tend);
%  PER4001_jja(:,:,tpointer+(0:tend-tstart)) = PER4001(:,:,tstart:tend);

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

  time_jja(tpointer+(0:tend-tstart)) = time(tstart:tend);
  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
hDJFstart = hours(datetime(yStart:yEnd-1,12,1,0,0,0)  - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
hDJFend   = hours(datetime(yStart+1:yEnd, 2,28,0,0,0) - datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
ds_djf=ds; ds_djf(3) = sum(hDJFend-hDJFstart)/24 +numel(hDJFstart); nd_djf=ds_djf(3)/(nyr-1);

mn2t_djf = zeros(ds_djf,'single');
mn2tCli_djf = zeros(ds_djf,'single');
WgtStar_djf = zeros(ds_djf,'single');
%Z500a_djf = zeros(ds_djf,'single');
%Cold_djf = false(ds_djf);
%PER0202_djf = false(ds_djf);
%PER2225_djf = false(ds_djf);
%PER4001_djf = false(ds_djf);
time_djf = zeros([ds_djf(3) 1], class(time));
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  mn2t_djf(:,:,tpointer+(0:tend-tstart)) = mn2t(:,:,tstart:tend);
  mn2tCli_djf(:,:,tpointer+(0:tend-tstart)) = mn2tCli(:,:,tstart:tend);
  WgtStar_djf(:,:,tpointer+(0:tend-tstart)) = mn2t_star(:,:,tstart:tend);
%  Z500a_djf(:,:,tpointer+(0:tend-tstart)) = Z500a(:,:,tstart:tend);
%  Cold_djf(:,:,tpointer+(0:tend-tstart)) = Cold(:,:,tstart:tend);  % old: all season

%  PER0202_djf(:,:,tpointer+(0:tend-tstart)) = PER0702(:,:,tstart:tend);
%  PER2225_djf(:,:,tpointer+(0:tend-tstart)) = PER2731(:,:,tstart:tend);
%  PER4001_djf(:,:,tpointer+(0:tend-tstart)) = PER4001(:,:,tstart:tend);

  time_djf(tpointer+(0:tend-tstart)) = time(tstart:tend);
  tpointer = tpointer +tend-tstart+1;
end
clear mx2t mn2t Z500a Hot Cold PER0202 PER2225 PER4001 PER0702 PER2731  yyyy tpointer tstart tend lat9 lon9 mat_z500a mat_load2 fn_load2 u v fc
disp('finish collect'); toc

%% x6xx: remove JJA mean
lsm_jja = lsm; lsm_jja(:,latt42(:)<=thresh{5})=false;
area_jja = lsm_jja*diag(areaEarth)/ds(1);
lsm_djf = lsm; lsm_djf(:,latt42(:)<=0)=false;
area_djf = lsm_djf*diag(areaEarth)/ds(1);
%lat_xy = repmat(latt42',[ds(1) 1]);
%lon_xy = repmat(lont42,[1 ds(2)]);
%nr=1; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(:,latt42>thresh{5},nr)=true; regArr{nr}='All >40N';
%nr=2; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(:,latt42>thresh{5},nr)=true; mask_xyr(:,:,nr)=mask_xyr(:,:,nr).*lsm_jja; regArr{nr}='Land >40N';
%nr=3; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=346|lont42<=191,:,nr)=true; mask_xyr(:,:,nr)=mask_xyr(:,:,nr).*lsm_jja; regArr{nr}='Eurasia';
%nr=4; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=191&lont42<=346,:,nr)=true; mask_xyr(:,:,nr)=mask_xyr(:,:,nr).*lsm_jja; regArr{nr}='N. America';
 %nr=3; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=30&lont42<=60,latt42>=45&latt42<=75,nr)=true; regArr{nr}='Russia';
 %nr=4; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=0&lont42<=30,latt42>=45&latt42<=75,nr)=true; regArr{nr}='Europe';
nr=1; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(:,latt42>40,nr)=true; regArr{nr}='All >40N';
nr=2; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(:,latt42>40,nr)=true; mask_xyr(:,:,nr)=mask_xyr(:,:,nr).*lsm_jja; regArr{nr}='Land >40N';
nr=3; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=346|lont42<=191,latt42>40,nr)=true; mask_xyr(:,:,nr)=mask_xyr(:,:,nr).*lsm_jja; regArr{nr}='Eurasia >40N';
nr=4; mask_xyr(:,:,nr) = false(ds(1:2)); mask_xyr(lont42>=191&lont42<=346,latt42>40,nr)=true; mask_xyr(:,:,nr)=mask_xyr(:,:,nr).*lsm_jja; regArr{nr}='N. America >40N';

mx2t_xyn = squeeze(mean( reshape(mx2tCli_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
mn2t_xyn = squeeze(mean( reshape(mn2tCli_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr
%{
% remove trend
mx2t_xyn = squeeze(mean( reshape(mx2t_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
mn2t_xyn = squeeze(mean( reshape(mn2t_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr

%
lsm_jja = lsm &(mean(mx2t_xyn,3)>0);  % found some place have JJA mean temperature smaller than all season mean
lsm_djf = lsm &(mean(mn2t_xyn,3)>0);  % basically northern hemisphere

mx2t_jja = mx2t_jja - reshape(repmat(reshape(movmean(mx2t_xyn,5,3), [ds(1:2) 1 nyr]),[1 1 ds_jja(3)/nyr 1]),ds_jja);
mn2t_djf = mn2t_djf - reshape(repmat(reshape(movmean(mn2t_xyn,5,3), [ds(1:2) 1 nyr-1]),[1 1 ds_djf(3)/(nyr-1) 1]),ds_djf);
% end x6xx
%}
disp('finish xyn'); toc

%area_jja = lsm_jja;
mx2tsd = std(double(mx2t_jja),[],3);
%sdJJA_NH = sqrt(areaEarth(yN1:yN2)*mean(mx2tsd(:,[yN1:yN2]).^2)'/sum(areaEarth(yN1:yN2)));
sdJJA_NH = sqrt( sum(area_jja(:).*mx2tsd(:).^2)./sum(area_jja(:)) );  % 2.9690
%mean( mx2tsd(lsm_jja)), sqrt(mean( mx2tsd(lsm_jja).^2))  %x909: 2.8920, 2.9750
mn2tsd = std(double(mn2t_djf),[],3);
%sdDJF_NH = sqrt(areaEarth(yN1:yN2)*mean(mn2tsd(:,[yN1:yN2]).^2)'/sum(areaEarth(yN1:yN2)));
sdDJF_NH = sqrt( sum(area_djf(:).*mn2tsd(:).^2)./sum(area_djf(:)) );  % 4.2022
%mean( mn2tsd(lsm_djf)), sqrt(mean( mn2tsd(lsm_djf).^2))  %x909: 4.2962, 4.7779
%mx2t_ns = permute(mx2t_jja,[3 1 2]);
%mn2t_ns = permute(mn2t_djf,[3 1 2]);
%sdJJA_NH = std(reshape( double(mx2t_ns(:,lsm_jja)) ,[],1));  % 2.9746
%sdJJA_SH = std(reshape( double(Wgt_jja(:,[yS1:yS2],:)) ,[],1));
%sdDJF_NH = std(reshape( double(mn2t_ns(:,lsm_djf)) ,[],1));  % 4.7773
%sdDJF_SH = std(reshape( double(Wgt_djf(:,[yS1:yS2],:)) ,[],1));
clear mx2t_ns mn2t_ns
disp(sprintf('jjaNH:%.2f djfNH:%.2f',sdJJA_NH,sdDJF_NH));

if (thresh{2}(1)=='q')
  HotQuantile = quantile(mx2t_jja, 1-thresh{1}, 3);
  ColdQuantile = quantile(mn2t_djf, 1-thresh{1}, 3);
elseif (thresh{2}(1)=='l')
  HotQuantile = prm.A1* mx2tsd;
  ColdQuantile = prm.A1* mn2tsd;
elseif (thresh{2}(1)=='s')
  HotQuantile = prm.A1* sdJJA_NH*ones(ds(1:2));
  ColdQuantile = prm.A1* sdDJF_NH*ones(ds(1:2));
else
  disp('Please enter quantile or sigma. Exitting..')
  return;
end
disp('finish quantile'); toc
Hot_jja = ( mx2t_jja> repmat( HotQuantile, [1 1 ds_jja(3)]) );
Cold_djf = ( mn2t_djf> repmat( ColdQuantile, [1 1 ds_djf(3)]) );

%  Hot_jja( abs(mx2t_jja) <thresh{4} ) =false;
%  Cold_djf( abs(mn2t_djf) <thresh{4} ) =false;
  Hot_jja( (mx2t_jja) <thresh{4} ) =false;
  Cold_djf( (mn2t_djf) <thresh{4} ) =false;
disp('finish t4'); toc

  Hot_jja = Hot_jja & repmat(lsm_jja, [1 1 ds_jja(3)]);
  Cold_djf = Cold_djf & repmat(lsm_djf, [1 1 ds_djf(3)]);
disp('finish lsm'); toc

sprintf('%.fK=%.1f%%=%.1fstd, final occurrence = %.1f%%',thresh{4}, 100*sum(sum(area_jja.*mean(mx2t_jja>thresh{4},3)))./sum(area_jja(:)), thresh{4}/sdJJA_NH, 100*sum(sum(area_jja.*mean(Hot_jja,3)))./sum(area_jja(:)) )

Hot_xyn = squeeze(mean( reshape(Hot_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr
%

%disp('return'); return;
%% label contiguous, see blocklabel, m2d
% method array
%Hotid_jja = repmat(reshape([1:ds(1)*floor(ds(2)/2)*nd_jja], [ds(1) floor(ds(2)/2) nd_jja]), [1 2 nyr]);
Hotid_jja = reshape([1:prod(ds_jja)], [ds_jja]);
Hotid_jja = Hotid_jja.*Hot_jja;
wrk = zeros(ds(1),floor(ds(2)/2)+1,nd_jja+1);
for yyyy = yStart:yEnd
  wrk(:,1:end-1,1:end-1) = Hotid_jja(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart)*nd_jja+(1:nd_jja));
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
  Hotid_jja(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart)*nd_jja+(1:nd_jja)) = wrk(:,1:end-1,1:end-1);
end
Hotid_jja(:,1:floor(ds(2)/2),:) = 0;

Coldid_djf = reshape([1:prod(ds_djf)], [ds_djf]);
Coldid_djf = Coldid_djf.*Cold_djf;
wrk = zeros(ds(1),floor(ds(2)/2)+1,nd_djf+1);
for yyyy = yStart+1:yEnd
  wrk(:,1:end-1,1:end-1) = Coldid_djf(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart-1)*nd_djf+(1:nd_djf));
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
  Coldid_djf(:,end:-1:end-floor(ds(2)/2)+1,(yyyy-yStart-1)*nd_djf+(1:nd_djf)) = wrk(:,1:end-1,1:end-1);
end
Coldid_djf(:,1:floor(ds(2)/2),:) = 0;
disp('finish label'); toc

clear wrk bwrk
clear mx2t_bar mx2t_star mx2t_hat mn2t_bar mn2t_star mn2t_hat tArr dtArr DArr mx2tCli mn2tCli
%clear WgtStar_jja WgtStar_djf mx2tCli_jja mn2tCli_djf  % QC
%save(['wrk_',ver,'.mat'],'-v7.3');  %TODO
%disp('finish save wrk'); toc
%disp('pause'); pause;
%

%{
clear;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
ver=['x919_',caseid];  %
load(['wrk_',ver,'.mat']);  %TODO
disp('finish load wrk'); toc
%}

%
Hotid_jja = categorical(Hotid_jja);
Hotid_jja = removecats(Hotid_jja,'0');
Hotid_jja = renamecats(Hotid_jja, cellstr(num2str([1:numel(categories(Hotid_jja))]')) );
HotAttr.id = categories(Hotid_jja);
nCat = numel(HotAttr.id);
wrk_ytc = permute(single(countcats(Hotid_jja,1)),[2 3 1]); %ytc
area_tc = reshape(areaEarth/ds(1) * wrk_ytc(:,:),[ds_jja(3) nCat]);
disp('a5'); toc
HotAttr.area = sum(area_tc,1)';
[~,HotAttr.tmax] = max(area_tc',[],2);
HotAttr.yr = ceil(HotAttr.tmax/nd_jja) +yStart-1;
HotAttr.tstart = zeros([nCat 1],'single');
HotAttr.tend = zeros([nCat 1],'single');

wrk_xyc = reshape( countcats(Hotid_jja,3), ds(1)*ds(2),nCat);
F_y      = repmat(reshape(1:ds(2), [1 ds(2)]), [ds(1) 1]) *diag(areaEarth)/ds(1);
F_repmat = repmat(reshape(1:ds(1), [ds(1) 1]), [1 ds(2)]);
F_sin    = sin( F_repmat*2*pi/ds(1)) *diag(areaEarth);  % new, weight by area
F_cos    = cos( F_repmat*2*pi/ds(1)) *diag(areaEarth);
HotAttr.x = atan2(F_sin(:)'*wrk_xyc, F_cos(:)'*wrk_xyc)'/2/pi*ds(1);
HotAttr.x = 0.5 + mod(HotAttr.x-0.5, ds(1));  % range from 0.5 - ds(1).5
HotAttr.x = 0.5 + mod(HotAttr.x-0.5, ds(1));  % bug..
HotAttr.y = (F_y(:)'*wrk_xyc)'./HotAttr.area;

wrk_xct = countcats(Hotid_jja,2);
HotAttr.xt = reshape(atan2(F_sin(:,end)'*wrk_xct(:,:), F_cos(:,end)'*wrk_xct(:,:)),[nCat ds_jja(3)])'/2/pi*ds(1);
HotAttr.xt = 0.5 + mod(HotAttr.xt-0.5, ds(1));  % range from 0.5 - ds(1).5
HotAttr.xt = 0.5 + mod(HotAttr.xt-0.5, ds(1));  % bug
HotAttr.yt = reshape(F_y(1,:)*wrk_ytc(:,:),[ds_jja(3) nCat])./area_tc;
HotAttr.xt(isnan(HotAttr.yt)) = nan;
%HotAttr.ut = %TODO
disp('before loop'); toc

HotAttr.areat = cell([nCat 1]);
for nBlock = 1:nCat
%  HotAttr.areat{nBlock} = area_tc(area_tc(:,nBlock+1)~=0, nBlock+1);
%  HotAttr.tstart(nBlock) = find(area_tc(:,nBlock+1),1);
  HotAttr.areat{nBlock} = area_tc(area_tc(:,nBlock)~=0, nBlock);
  HotAttr.tstart(nBlock) = find(area_tc(:,nBlock),1);
  HotAttr.tend(nBlock) = find(area_tc(:,nBlock),1,'last');
end

Coldid_djf = categorical(Coldid_djf);
Coldid_djf = removecats(Coldid_djf,'0');
Coldid_djf = renamecats(Coldid_djf, cellstr(num2str([1:numel(categories(Coldid_djf))]')) );
ColdAttr.id = categories(Coldid_djf);
nCat = numel(ColdAttr.id);
wrk_ytc = permute(single(countcats(Coldid_djf,1)),[2 3 1]); %ytc
area_tc = reshape(areaEarth/ds(1) * wrk_ytc(:,:),[ds_djf(3) nCat]);
disp('b5'); toc
ColdAttr.area = sum(area_tc,1)';
[~,ColdAttr.tmax] = max(area_tc',[],2);
ColdAttr.yr = ceil(ColdAttr.tmax/nd_djf) +yStart;
ColdAttr.tstart = zeros([nCat 1],'single');
ColdAttr.tend = zeros([nCat 1],'single');

wrk_xyc = reshape( countcats(Coldid_djf,3), ds(1)*ds(2),nCat);
%F_y      = repmat(reshape(1:ds(2), [1 ds(2)]), [ds(1) 1]) *diag(areaEarth)/ds(1);
%F_repmat = repmat(reshape(1:ds(1), [ds(1) 1]), [1 ds(2)]);
%F_sin    = sin( F_repmat*2*pi/ds(1)) *diag(areaEarth);  % new, weight by area
%F_cos    = cos( F_repmat*2*pi/ds(1)) *diag(areaEarth);
ColdAttr.x = atan2(F_sin(:)'*wrk_xyc, F_cos(:)'*wrk_xyc)'/2/pi*ds(1);
ColdAttr.x = 0.5 + mod(ColdAttr.x-0.5, ds(1));  % range from 0.5 - ds(1).5
ColdAttr.x = 0.5 + mod(ColdAttr.x-0.5, ds(1));  % bug..
ColdAttr.y = (F_y(:)'*wrk_xyc)'./ColdAttr.area;

wrk_xct = countcats(Coldid_djf,2);
ColdAttr.xt = reshape(atan2(F_sin(:,end)'*wrk_xct(:,:), F_cos(:,end)'*wrk_xct(:,:)),[nCat ds_djf(3)])'/2/pi*ds(1);
ColdAttr.xt = 0.5 + mod(ColdAttr.xt-0.5, ds(1));  % range from 0.5 - ds(1).5
ColdAttr.xt = 0.5 + mod(ColdAttr.xt-0.5, ds(1));  % bug
ColdAttr.yt = reshape(F_y(1,:)*wrk_ytc(:,:),[ds_djf(3) nCat])./area_tc;
ColdAttr.xt(isnan(ColdAttr.yt)) = nan;
%ColdAttr.ut = %TODO
disp('before loop'); toc

ColdAttr.areat = cell([nCat 1]);
for nBlock = 1:nCat
%  ColdAttr.areat{nBlock} = area_tc(area_tc(:,nBlock+1)~=0, nBlock+1);
%  ColdAttr.tstart(nBlock) = find(area_tc(:,nBlock+1),1);
  ColdAttr.areat{nBlock} = area_tc(area_tc(:,nBlock)~=0, nBlock);
  ColdAttr.tstart(nBlock) = find(area_tc(:,nBlock),1);
  ColdAttr.tend(nBlock) = find(area_tc(:,nBlock),1,'last');
end

clear wrk_ytc wrk_xyc wrk_xct F_y F_repmat F_sin F_cos
clear area_tc
%

%{
nBlock = 0;
idUniq = unique(Hotid_jja(:));
HotAttr.id = idUniq(idUniq>0);
HotAttr.areat = cell([nnz(idUniq) 1]);
HotAttr.area = zeros([nnz(idUniq) 1],'single');
for id = idUniq(idUniq>0).'
  nBlock = nBlock +1;
%  HotAttr.id(nBlock) = id;
  wrk = squeeze(mean( (Hotid_jja==id) ,1));
  wrk_t = ( areaEarth * wrk )';
  HotAttr.areat{nBlock} = wrk_t(wrk_t~=0);
  HotAttr.area(nBlock) = sum(HotAttr.areat{nBlock});

%       % for cyclic boundary condition, find xRef (labels dont cross) to be boundary
%       xRef = find( all(F_id(:,:,t)~=id, 2), 1);
%       if (isempty(xRef))
%           disp(['t=',num2str(t), ', id=',num2str(id)])
%           continue
%       end
%       F_x = xRef + mod(F_repmat-xRef, ds(1));
%       tmp = [t; id; mean(F_x(F_id(:,:,t)==id)); mean(F_y(F_id(:,:,t)==id))];
%         % un-weighted average of x,y index
end
%}
disp('finish attr'); toc

[~,csort] = sort(HotAttr.area,'descend');
%HotAttr.areat{csort(1)}
%HotAttr.tmax(csort(1))
%HotAttr.tstart(csort(1))
%HotAttr.x(csort(1))
%HotAttr.y(csort(1))
%HotAttr.xt(~isnan(HotAttr.yt(:,csort(1))),csort(1))
%HotAttr.yt(~isnan(HotAttr.yt(:,csort(1))),csort(1))

strTitleX=strTitle;
%save(['temp_',textTH,'_',text,'.mat'],'-v7.3');  % TODO
save(['temp_',ver,'.mat'],'-v7.3');  % TODO
disp('finish save'); toc
%disp('pause'); pause;
%

%toc
%% quality check
% meanZ500 in bindex_pre_intp_pchan
% meanJJA in xtrm_reanalysis_pchan02.m
addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added
rng(2017);

fn_figure = ['../index_wise/all',ver,'.ps'];
system(['rm ',fn_figure]);

%
Wgtjja_xyn = mx2t_xyn;
Wgtdjf_xyn = mn2t_xyn;
jjaQuantile = HotQuantile;
djfQuantile = ColdQuantile;
Wgt_jja = mx2t_jja;
Wgt_djf = mn2t_djf;


% xyn trend p.1
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
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','eqdazim','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','breusing','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

Wgtjja_trend(~lsm_jja) = nan;
%contourfm(latt42,lonplot,double(Wgtjja_trend([1:end,1],:)'),[-4:4]*2e-2);  % TODO
pcolormPC(latt42,lont42, 30*Wgtjja_trend'); shading flat;
%contourm(latt42,lonplot,double(Wgtjja_trend([1:end,1],:)'),[-3:3]*2e-2); %axis equal tight; %,'showtext','on'
%contourm(latt42,lonplot,double(Wgtjja_trend([1:end,1],:)'),'showtext','on'); %axis equal tight; %,'showtext','on'
%colormap(gca,jet(12)); caxis([0 12]); colorbar;
%colormap(gca,b2rPC(-5*2e-2, 5*2e-2, 10)); colorbar;  %TODO
colormap(gca,b2rPC(-5*0.5, 5*0.5, 10)); colorbar;  %TODO
plotm(coastlat,coastlon,'k')
title({strTitle,'JJA Wgt trends (1979-2017) (unit/30yr)'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

Wgtdjf_trend(~lsm_djf) = nan;
%contourfm(latt42,lonplot,double(Wgtdjf_trend([1:end,1],:)'),[-4:4]*2e-2);
pcolormPC(latt42,lont42, 30*Wgtdjf_trend'); shading flat;
%contourm(latt42,lonplot,double(Wgtdjf_trend([1:end,1],:)'),[-0.06:0.02:0.06]); %axis equal tight;
%colormap(gca,jet(12)); caxis([0 12]); colorbar;
%colormap(gca,b2rPC(-5*2e-2, 5*2e-2, 10)); colorbar;
colormap(gca,b2rPC(-5*0.5, 5*0.5, 10)); colorbar;
plotm(coastlat,coastlon,'k')
title({'DJF Wgt trends (1979/80-2016/17) (unit/30yr)'},'fontsize',16);
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);
%

% xyn: mean JJA/DJF time series for ramdom points: check movmean 5
%{
jArr = find(latt42>0); jArr = jArr(1:2:end); %set(groot,'defaultAxesColorOrder',hsv(length(jArr)));
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);

subplot(1,2,1); hold on;
xlabel({'year'}); ylabel('Wgt (unit): each line shifted by 3');  %TODO
ax=gca; ax.ColorOrder = hsv(length(jArr));
%ax.ColorOrder = lines(length(jArr));
%for j=jArr(:)'
for jj = 1:length(jArr)
  j = jArr(jj); 
  nLand = sum(lsm_jja(:,j));
  if (nLand>0) 
    iArr = find(lsm_jja(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    ax.ColorOrderIndex = jj;
%    plot(yStart:yEnd, squeeze(Wgtjja_xyn(i,j,:)));
%    ax.ColorOrderIndex = jj;
%    plot(yStart:yEnd, movmean( squeeze(Wgtjja_xyn(i,j,:)), 5), '--');
    plot(yStart:yEnd, jj*3-mean(Wgtjja_xyn(i,j,1:3)) +squeeze(Wgtjja_xyn(i,j,:)));
    ax.ColorOrderIndex = jj;
    plot(yStart:yEnd, jj*3-mean(Wgtjja_xyn(i,j,1:3)) +movmean( squeeze(Wgtjja_xyn(i,j,:)), 5), '--');
  end  % TODO land or not
end
title({strTitle,'mean Wgt in JJA'});
xlim([yStart yEnd]);

subplot(1,2,2); hold on;
xlabel({'year'}); ylabel('Wgt (unit): each line shifted by 3');
ax=gca; ax.ColorOrder = hsv(length(jArr));
%ax.ColorOrder = lines(length(jArr));
%for j=jArr(:)'
for jj = 1:length(jArr)
  j = jArr(jj); 
  nLand = sum(lsm_djf(:,j));
  if (nLand>0) 
    iArr = find(lsm_djf(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    ax.ColorOrderIndex = jj;
%    plot(yStart+1:yEnd, squeeze(Wgtdjf_xyn(i,j,:)));
%    ax.ColorOrderIndex = jj;
%    plot(yStart+1:yEnd, movmean( squeeze(Wgtdjf_xyn(i,j,:)), 5), '--');
    plot(yStart+1:yEnd, jj*3-mean(Wgtdjf_xyn(i,j,1:3)) +squeeze(Wgtdjf_xyn(i,j,:)));
    ax.ColorOrderIndex = jj;
    plot(yStart+1:yEnd, jj*3-mean(Wgtdjf_xyn(i,j,1:3)) +movmean( squeeze(Wgtdjf_xyn(i,j,:)), 5), '--');
  end
end
title('mean Wgt in DJF');

print(gcf, '-dpsc2','-append',fn_figure);
%}

%% x9xx: SeasonalCycle, Wgt, xtrm_reanalysis: check 15/15day
rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
jArr = find(latt42>0); jArr = jArr(1:2:end);
%jArr = [yN1:yN2]';
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_jja(:,j));
  if (nLand>0)
    iArr = find(lsm_jja(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    subplot(3,5,jj); hold on;
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr);
    plot(reshape( squeeze(WgtStar_jja(i,j,:)),[],nyr ));
    plot(mean(reshape( squeeze(WgtStar_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%    plot([1 nd_jja],prm.A*[1 1],'k-');  % TODO
%    plot([1 nd_jja],jjaQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([sprintf('%.0fN %.0fE',latt42(j),lont42(i))]);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
%print(gcf,'-dpdf','-r600',['SeasonalCycleJJA_',textTH,'_',text,'.pdf'])

rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
jArr = find(latt42>0); jArr = jArr(1:2:end);
%jArr = [yN1:yN2]';
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_djf(:,j));
  if (nLand>0)
    iArr = find(lsm_djf(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    subplot(3,5,jj); hold on;
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr-1);
    plot(reshape( squeeze(WgtStar_djf(i,j,:)),[],nyr-1 ));
    plot(mean(reshape( squeeze(WgtStar_djf(i,j,:)),[],nyr-1 ),2),'k','linewidth',1);
%    plot([1 90],prm.A*[1 1],'k-');  % TODO
%    plot([1 90],djfQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([sprintf('%.0fN %.0fE',latt42(j),lont42(i))]);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
%print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])

%% x9xx: SeasonalCycle, Wgt, xtrm_reanalysis: check 11yr, p.4-5
rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
jArr = find(latt42>0); jArr = jArr(1:2:end);
%jArr = [yN1:yN2]';
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_jja(:,j));
  if (nLand>0)
    iArr = find(lsm_jja(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    subplot(3,5,jj); hold on;
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr);
    plot(reshape( squeeze(mx2tCli_jja(i,j,:)),[],nyr ));
    plot(mean(reshape( squeeze(mx2tCli_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%    plot([1 nd_jja],prm.A*[1 1],'k-');  % TODO
%    plot([1 nd_jja],jjaQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([sprintf('%.0fN %.0fE',latt42(j),lont42(i))]);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
%print(gcf,'-dpdf','-r600',['SeasonalCycleJJA_',textTH,'_',text,'.pdf'])

rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
jArr = find(latt42>0); jArr = jArr(1:2:end);
%jArr = [yN1:yN2]';
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_djf(:,j));
  if (nLand>0)
    iArr = find(lsm_djf(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    subplot(3,5,jj); hold on;
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr-1);
    plot(reshape( squeeze(mn2tCli_djf(i,j,:)),[],nyr-1 ));
    plot(mean(reshape( squeeze(mn2tCli_djf(i,j,:)),[],nyr-1 ),2),'k','linewidth',1);
%    plot([1 90],prm.A*[1 1],'k-');  % TODO
%    plot([1 90],djfQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([sprintf('%.0fN %.0fE',latt42(j),lont42(i))]);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
%print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])

%% plot quantile xtrm_reanalysis_pchan02.m / xtrmfreq, p.6
%
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

jjaQuantile(~lsm_jja) = nan;
%contourfm(latt42,lonplot,double(jjaQuantile([1:end,1],:)'),[-2:0.5:2]);
pcolormPC(latt42,lont42,jjaQuantile'); shading flat;
colormap(gca,jet(12)); caxis([0 12]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitle,'JJA threshold'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

djfQuantile(~lsm_djf) = nan;
%contourfm(latt42,lonplot,double(djfQuantile([1:end,1],:)'),[-2:0.5:2]);
pcolormPC(latt42,lont42,djfQuantile'); shading flat;
colormap(gca,jet(12)); caxis([0 24]); colorbar; caxis auto;
plotm(coastlat,coastlon,'k')
title({'DJF threshold'},'fontsize',16);
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);
%

%% plot standard deviation xtrm_skew (1d) qc20180412
%{
% qc 20180806
addpath('/n/home05/pchan/bin');
load coastlines
caseid=['ERA-interim_19790101-20171231'];
verX='x919';
load(['temp_',verX,'_',caseid,'.mat']);
Wgt_jja = mx2t_jja;
jjaMom2 = std(Wgt_jja, [],3);
jjaMom2(~lsm_jja) = nan;
[iii,jjj]=max(jjaMom2(:)), [i,j]=ind2sub(ds(1:2),jjj), strT='coast';m=0; i=53;j=58; % sprintf('%.0fN %.0fE',latt42(j),lont42(i))

Hot_xydn = reshape(Hot_jja,[ds(1:2),nd_jja,nyr]);
%Hot_d = areaEarth*squeeze(mean(mean(Hot_xydn,4),1));
ccc_xy = mean(sum(Hot_xydn(:,:,:,2010-yStart+1),4),3);
%[iii,jjj]=max(ccc_xy(:)), [i,j]=ind2sub(ds(1:2),jjj), strT='2010';m=3; i=17; % sprintf('%.0fN %.0fE',latt42(j),lont42(i))
%i=10;j=50; strT='env';m=6; % sprintf('%.0fN %.0fE',latt42(j),lont42(i))

histc_mx2t = histcounts(mx2t_jja(i,j,:),[-20:0.1:20]);
if (sum(histc_mx2t)~=ds_jja(3)) disp('histc error'); return; end
figure(90);
set(gcf,'units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,2,1); hold on;
plot([-19.9:0.1:20],cumsum(histc_mx2t)/ds_jja(3) ,'DisplayName',[sprintf('%.0fN %.0fE std%.2fK',latt42(j),lont42(i),jjaMom2(i,j))]);
legend show; legend('location','southeast'); grid on; xlim([4 15]);
xlabel('Temperature anomaly (K)'); title('CDF');
set(gca,'fontsize',18);

subplot(2,2,3); hold on;
%plot([-19.9:0.2:19.9],(histc_mx2t)/ds_jja(3) ,'DisplayName',[sprintf('%.0fN %.0fE std%.2fK',latt42(j),lont42(i),jjaMom2(i,j))]);
plot([-19.75:0.5:19.75],sum(reshape(histc_mx2t,5,[]))/ds_jja(3) ,'DisplayName',[sprintf('%.0fN %.0fE std%.2fK',latt42(j),lont42(i),jjaMom2(i,j))]);
grid on; xlim([-15 15]);
xlabel('Temperature anomaly (K)'); title('PDF');
set(gca,'fontsize',18);

arrth = [4:0.2:15];
stdfreq=nan([numel(arrth) 1]);
for nn = 1:numel(arrth)
  stdfreq(nn) = std(mean(reshape(mx2t_jja(i,j,:)>arrth(nn),nd_jja,nyr),1),[],2);
end
subplot(2,2,2); hold on;
plot(arrth,stdfreq ,'DisplayName',[sprintf('%.0fN %.0fE std%.2fK',latt42(j),lont42(i),jjaMom2(i,j))]);
grid on; xlim([4 15]);
xlabel('Threshold (K)'); title('Interannual std of occurrence');
set(gca,'fontsize',18);

fn_figure = ['qc_',verX,'_20180806.ps'];
print(gcf, '-dpsc2','-append',fn_figure);
system(['ps2pdf ',fn_figure]);

%
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

%contourfm(latt42,lonplot,double(jjaMom2([1:end,1],:)'),[-2:0.5:2]);
pcolormPC(latt42,lont42,jjaMom2'); shading flat;
%colormap(gca,jet(12)); caxis([0 12]); colorbar; caxis auto;  % TODO
colormap(gca,colormap_CD([0.45 0.70; 0.25 0.9],[.7 .35],[0 0],6)); caxis([0 999]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
title({strTitle,'JJA std (K)'},'fontsize',16);
tightmap;

subplot(2,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

%contourfm(latt42,lonplot,double(jjaMom2([1:end,1],:)'),[-2:0.5:2]);
%pcolormPC(latt42,lont42,100*normcdf(-3*sdJJA_NH./jjaMom2)'); shading flat;
pcolormPC(latt42,lont42,100*normcdf(-5./jjaMom2)'); shading flat;
%colormap(gca,flipud(hot(10))); caxis([0 2.6733]); colorbar; %caxis auto;  % TODO
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],12)); caxis([0 15]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
title({strTitle,'JJA normcdf (%)'},'fontsize',16);
tightmap;

subplot(2,2,3);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

%PER_jja = mx2t_jja>5;
PERfreq_jja = mean(PER_jja,3);
PERfreq_jja(~lsm_jja) = nan;
%contourfm(latt42,lonplot,double(jjaMom2([1:end,1],:)'),[-2:0.5:2]);
pcolormPC(latt42,lont42,100*PERfreq_jja'); shading flat;
%colormap(gca,flipud(hot(10))); caxis([0 2.6733]); colorbar; %caxis auto;  % TODO
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],12)); caxis([0 15]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
title({strTitle,'JJA frequency (%)'},'fontsize',16);
tightmap;

subplot(2,2,4);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);

PERstd_jja = std(mean(reshape(PER_jja,[ds(1:2),nd_jja,nyr]),3),[],4);
PERstd_jja(~lsm_jja) = nan;
%contourfm(latt42,lonplot,double(jjaMom2([1:end,1],:)'),[-2:0.5:2]);
pcolormPC(latt42,lont42,100*PERstd_jja'); shading flat;
%colormap(gca,flipud(hot(10))); caxis([0 2.6733]); colorbar; %caxis auto;  % TODO
colormap(gca,colormap_CD([0.16 0.89],[1 0.3],[0],12)); caxis([0 999]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7])
title({strTitle,'JJA frequency std (%)'},'fontsize',16);
tightmap;


% qc 20180620
addpath('/n/home05/pchan/bin');
load coastlines
caseid=['ERA-interim_19790101-20171231'];
verX='x919';
load(['temp_',verX,'_',caseid,'.mat']);
Hot_xydn = reshape(Hot_jja,[ds(1:2),nd_jja,nyr]);
Hot_d = areaEarth*squeeze(mean(mean(Hot_xydn,4),1));
mmm_xy = mean(sum(Hot_xydn,4),3);
aaa_xy = mean(sum(Hot_xydn(:,:,1:8,:),4),3);
%bbb_xy = mean(sum(Hot_xydn(:,:,54:71,:),4),3);
[iii,jjj]=max(mmm_xy(:)), [i,j]=ind2sub(ds(1:2),jjj), strT='JJA';m=0;
[iii,jjj]=max(aaa_xy(:)), [i,j]=ind2sub(ds(1:2),jjj), strT='Day1-8';m=3;
%[iii,jjj]=max(bbb_xy(:)), [i,j]=ind2sub(ds(1:2),jjj), strT='Day54-71';m=6;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,2,1);
plot(Hot_d,'o-'); grid on;
xlabel({'day'}); title({'Mean JJA extreme area (km^2)'});
subplot(2,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
pcolormPC(latt42,lont42, mmm_xy'); colorbar; tightmap;
colormap(gca,flipud(hot(10))); plotm(coastlat,coastlon,'k'); title({'JJA extreme frequency (%)'},'fontsize',16);
subplot(2,2,3);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
pcolormPC(latt42,lont42, aaa_xy'); colorbar; tightmap;
colormap(gca,flipud(hot(10))); plotm(coastlat,coastlon,'k'); title({'Day1-8 extreme frequency (%)'},'fontsize',16);
%subplot(2,2,4);
%axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%pcolormPC(latt42,lont42, bbb_xy'); colorbar; tightmap;
%colormap(gca,flipud(hot(10))); plotm(coastlat,coastlon,'k'); title({'Day54-71 extreme frequency (%)'},'fontsize',16);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
% check 15/15day
subplot(3,3,m+1); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
plot(reshape( squeeze(WgtStar_jja(i,j,:)),[],nyr ));
plot(mean(reshape( squeeze(WgtStar_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
%title([sprintf('%.0fN %.0fE',latt42(j),lont42(i))]);
title({'Smoothed along day',[num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E chosen for ',strT]});
axis tight;

% check 11yr
subplot(3,3,m+2); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
plot(reshape( squeeze(mx2tCli_jja(i,j,:)),[],nyr ));
plot(mean(reshape( squeeze(mx2tCli_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
title({'Smoothed along day&year',[num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E chosen for ',strT]});
axis tight;

%
jjaQuantile = HotQuantile;
Wgt_jja = mx2t_jja;
subplot(3,3,m+3); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
plot(reshape( squeeze(Wgt_jja(i,j,:)),[],nyr ));
plot(mean(reshape( squeeze(Wgt_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%plot([1 nd_jja],prm.A*[1 1],'k-');  % TODO
plot([1 nd_jja],jjaQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
title({'Anomalies',[num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E chosen for ',strT]});
axis tight;

fn_figure = ['qc_',verX,'_20180620.ps'];
print(gcf, '-dpsc2','-append',fn_figure);
system(['ps2pdf ',fn_figure]);
%}


%% SeasonalCycle, Wgt, xtrm_reanalysis
%
rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
jArr = find(latt42>0); jArr = jArr(1:2:end);
%jArr = [yN1:yN2]';
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_jja(:,j));
  if (nLand>0)
    iArr = find(lsm_jja(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    subplot(3,5,jj); hold on;
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr);
    plot(reshape( squeeze(Wgt_jja(i,j,:)),[],nyr ));
    plot(mean(reshape( squeeze(Wgt_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%    plot([1 nd_jja],prm.A*[1 1],'k-');  % TODO
    plot([1 nd_jja],jjaQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([sprintf('%.0fN %.0fE',latt42(j),lont42(i))]);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);
%print(gcf,'-dpdf','-r600',['SeasonalCycleJJA_',textTH,'_',text,'.pdf'])

rng(2017);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
jArr = find(latt42>0); jArr = jArr(1:2:end);
%jArr = [yN1:yN2]';
for jj = 1:length(jArr)
  j = jArr(jj);
  nLand = sum(lsm_djf(:,j));
  if (nLand>0)
    iArr = find(lsm_djf(:,j));
    i = iArr(randi(nLand));
%    i = randi(ds(1));
    subplot(3,5,jj); hold on;
    ax=gca; ax.ColorOrder = jet(nyr-1);
    plot(reshape( squeeze(Wgt_djf(i,j,:)),[],nyr-1 ));
    plot(mean(reshape( squeeze(Wgt_djf(i,j,:)),[],nyr-1 ),2),'k','linewidth',1);
%    plot([1 90],prm.A*[1 1],'k-');
    plot([1 90],djfQuantile(i,j)*[1 1],'k-');
    title([sprintf('%.0fN %.0fE',latt42(j),lont42(i))]);
    axis tight;
  end
end
print(gcf, '-dpsc2','-append',fn_figure);
%print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])
%


%system(['ps2pdf ',fn_figure]);
toc
%


%% quick polyfit, lagcorr
%
 %text='AOefbCV1HF'; thresh={0.01,'quantile',5,5,0}; caseid=[text,'T63h00'];
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
 %load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','ds','Cold_yxht','Hot_yxht','T850f_yxht','areaEarth')
%load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','yStart','yEnd','nyr' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend')

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
Wgt_jja = mx2t_jja;
Wgt_djf = mn2t_djf;
PER_jja = Hot_jja;
PER_djf = Cold_djf;
PERjjaAttr = HotAttr;
PERid_jja = Hotid_jja;
PERdjfAttr = ColdAttr;
PERid_djf = Coldid_djf;

%timeNan(timeNan>ds(3)) = [];

%% collect JJA
%{
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
%}

%%
 %min(min(min(mx2t_jja(Hot_jja))))
mx2t_jja = mx2t_jja - repmat(max(HotQuantile,thresh{4}),[1 1 ds_jja(3)]);
%Wgt_jja = Wgt_jja - prm.A* repmat([sdJJA_SH*ones(1,ds(2)/2), sdJJA_NH*ones(1,ds(2)/2)],[ds(1) 1 ds_jja(3)]);
Wgt_jja = mx2t_jja;

Hot_n = [areaEarth * squeeze(mean(mean(reshape(Hot_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
Hotw_n = [areaEarth * squeeze(mean(mean(reshape(Hot_jja.*mx2t_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERwjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*Wgt_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERljja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*lsm_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERlwjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*lsm_jja.*Wgt_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
disp(sprintf('jja: %+.3f %+.3f %+.3f %+.3f', corr(PERjja_n(:),Hot_n(:)), corr(PERljja_n(:),Hot_n(:)) , corr(PERwjja_n(:),Hotw_n(:)), corr(PERlwjja_n(:),Hotw_n(:)) ));
%disp(sprintf('jja: prst=%.2f A=%.1f %+.3f %+.3f %+.3f %+.3f',prm.prst,prm.A, corr(PERjja_n(:),Hot_n(:)), corr(PERljja_n(:),Hot_n(:)) , corr(PERwjja_n(:),Hotw_n(:)), corr(PERlwjja_n(:),Hotw_n(:)) ));
%disp(sprintf('jja: Davg0=%i A=%.1f %+.3f %+.3f',prm.Davg0,prm.A, corr(PERjja_n(:),Hot_n(:)), corr(PERljja_n(:),Hot_n(:))));

wrkar1 = corr(Hot_n(2:end),Hot_n(1:end-1));
wrkar2 = corr(PERljja_n(2:end),PERljja_n(1:end-1));
wrkar3 = corr(PERjja_n(2:end),PERjja_n(1:end-1));
disp(sprintf('autocorr H:%+.3f Pl:%+.3f P:%+.3f tstar/t:%.3f %.3f kurt:%.1f',wrkar1,wrkar2,wrkar3,(1-wrkar1*wrkar2)/(1+wrkar1*wrkar2) ,(1-wrkar1*wrkar3)/(1+wrkar1*wrkar3) ,kurtosis(PERljja_n) ));
wrkar1 = corr(Hotw_n(2:end),Hotw_n(1:end-1));
wrkar2 = corr(PERlwjja_n(2:end),PERlwjja_n(1:end-1));
wrkar3 = corr(PERwjja_n(2:end),PERwjja_n(1:end-1));
disp(sprintf('autocorr H:%+.3f Pl:%+.3f P:%+.3f tstar/t:%.3f %.3f kurt:%.1f',wrkar1,wrkar2,wrkar3,(1-wrkar1*wrkar2)/(1+wrkar1*wrkar2) ,(1-wrkar1*wrkar3)/(1+wrkar1*wrkar3) ,kurtosis(PERlwjja_n) ));

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

save(['../index_wise/BlockFreq_',ver,'.mat'],'ver','strTitle','prm','timeNan','lont42','latt42','PERid_jja','PERjjaAttr','PERid_djf','PERdjfAttr','PER_jja','PER_djf','Wgt_jja','Wgt_djf', 'Hot_n','Hotw_n','PERjja_n','PERwjja_n','PERljja_n','PERlwjja_n','-v7.3');
system(['ln -s BlockFreq_',ver,'.mat ../index_wise/BlockFreq_',verX,ver,'.mat']);
%disp('pause');pause
% 'time','PER','Wgt',


%% SeasonalCycle, Area in xtrm_reanalysis_pchan02.m
%
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%load(['temp_',textTH,'_',text,'.mat'], 'nyr',  'yStart','yEnd')

textWgtArr = {'org','wgt'};
for textTH = textWgtArr

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
%textTH = [textTH,'_',textWgt{1}];
textTH = textTH{1};
if (contains(textTH, 'wgt'))
  PERjja_t = [areaEarth * squeeze(mean(PER_jja.*Wgt_jja,1))]';
  PERdjf_t = [areaEarth * squeeze(mean(PER_djf.*Wgt_djf,1))]';
else
  PERjja_t = [areaEarth * squeeze(mean(PER_jja,1))]';
  PERdjf_t = [areaEarth * squeeze(mean(PER_djf,1))]';
end

%ver=['abcd_',caseid];
%  load(['../index_wise/scatter_',textTH,'_',ver,'.mat'])
subplot(2,2,1); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
if (contains(textTH, 'wgt'))
  xlabel({'day'}); ylabel({'Weighted extreme area (unit*km^2)'});
else
  xlabel({'day'}); ylabel({'Extreme area (km^2)'});
end
plot(reshape( PERjja_t,[],nyr ));
plot(mean(reshape( PERjja_t,[],nyr ),2),'k','linewidth',1);
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
ylim([0 max(PERjja_t(:))]); grid on;
title([strTitle,': JJA']);
set(gca,'fontsize',12);

subplot(2,2,3); hold on;
ax=gca; ax.ColorOrder = jet(nyr);
if (contains(textTH, 'wgt'))
  xlabel({'year'}); ylabel({'Mean weighted extreme area (unit*km^2)'});
else
  xlabel({'year'}); ylabel({'Mean extreme area (km^2)'});
end
plot([1;1]*[yStart:yEnd], [zeros(1,nyr); mean(reshape(PERjja_t(:),[],nyr),1)]);
tmpstat = [polyfit((yStart:yEnd),mean(reshape(PERjja_t(:),[],nyr),1) ,1)];
fplot(@(x) polyval(tmpstat,x), [yStart yEnd], 'k');
title(['Legend, r=',num2str(corr((yStart:yEnd)',mean(reshape(PERjja_t(:),[],nyr),1)'),'%+.2f')]);
xlim([yStart yEnd]); ylim([0 max(mean(reshape(PERjja_t(:),[],nyr),1))]); grid on;
set(gca,'fontsize',12);


subplot(2,2,2); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
if (contains(textTH, 'wgt'))
  xlabel({'day'}); ylabel({'Weighted extreme area (unit*km^2)'});
else
  xlabel({'day'}); ylabel({'Extreme area (km^2)'});
end
plot(reshape( PERdjf_t,[],nyr-1 ));
plot(mean(reshape( PERdjf_t,[],nyr-1 ),2),'k','linewidth',1);
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
ylim([0 max(PERdjf_t(:))]); grid on;
title([strTitle,': DJF']);
set(gca,'fontsize',12);

subplot(2,2,4); hold on;
ax=gca; ax.ColorOrder = jet(nyr-1);
if (contains(textTH, 'wgt'))
  xlabel({'year'}); ylabel({'Mean weighted extreme area (unit*km^2)'});
else
  xlabel({'year'}); ylabel({'Mean extreme area (km^2)'});
end
plot([1;1]*[yStart+1:yEnd], [zeros(1,nyr-1); mean(reshape(PERdjf_t(:),[],nyr-1),1)]);
tmpstat = [polyfit((yStart+1:yEnd),mean(reshape(PERdjf_t(:),[],nyr-1),1) ,1)];
fplot(@(x) polyval(tmpstat,x), [yStart yEnd], 'k');
title(['Legend, r=',num2str(corr((yStart+1:yEnd)',mean(reshape(PERdjf_t(:),[],nyr-1),1)'),'%+.2f')]);
xlim([yStart+1 yEnd]); ylim([0 max(mean(reshape(PERdjf_t(:),[],nyr-1),1))]); grid on;
set(gca,'fontsize',12);

%savefig(gcf,['SeasonalCycleJJAareas_',textTH,'_',text,'.fig'])
%print(gcf,'-dpdf',['SeasonalCycleJJAareas_',textTH,'_',text,'.pdf'])
print(gcf, '-dpsc2','-append',fn_figure);
end % textWgt loop
%

system(['ps2pdf ',fn_figure]);
system(['rm ',fn_figure]);



%% TODO run all indices
%disp('return'); return;
%cd ../index_wise
%addpath('../matlab');  % suppress warning loading temp.mat

clearvars -except  verX caseid;
prm.S = 2.5e6;
prm.R = 1;
prm.D = 5;
prm.A = 1.5;
strTitle = sprintf('0a13: D13, psi500, D=%i,astd40-90N,S=%g,A=%g 15/15day*11yr',prm.D,prm.S,prm.A);
BlockStat_d2d_pchan06


fn_figure = ['corrTable_',verX,'.ps'];
system(['rm ',fn_figure]);

%  verX='x919';
%  caseid=['ERA-interim_19790101-20171231'];
%load(['../index_wise/tune_',verX,'2axx_',caseid,'.mat'],'rrrp'); rrr=rrrp;
%load(['../index_wise/BlockFreq_',verX,'2a73_',caseid,'.mat'],'Hot_n','PERljja_n','PERjja_n');
%{
nyr = 39;
Hot_n = normrnd(0,1,[nyr 1]);
rrr.pArr = 1:100; rrr.qArr = 1;
rrr.PERljja_npq = 0.6*repmat(Hot_n/std(Hot_n),[1 numel(rrr.pArr)]) + 0.8*normrnd(0,1,[nyr numel(rrr.pArr)]);

% clearvars -except  verX caseid Hot_n Hot_xyn;
addpath('/n/home05/pchan/bin');
  verX='x919';
  caseid=['ERA-interim_19790101-20171231'];
%load(['../index_wise/tune_',verX,'2axx_',caseid,'.mat'],'rrrd'); rrr=rrrd;
load(['../index_wise/BlockFreq_',verX,'2a75_',caseid,'.mat'],'Hot_n');
load(['temp_',verX,'_',caseid,'.mat'],'Hot_xyn');
nyr = length(Hot_n);

 % :1946,2038s/sqrt//gce | exe '1946,2038g!/ytest)/s/\.^2//gce' | 1946,2038g!/ytest)/s/\([^.]\)\^2/\1/gce   | 1946,2038s/ytest) sqrt/ytest) /gce | 1946,2038g/sqrt/s/\.^2//gce
 % :1946,2038s/FontWeight','bold/Color','m/gce | exe '1946,2038g/HorizontalAlignment/s/1f/2f/gce' | exe '1946,2038g/pcolorPH/s/);$/& shading faceted;/gce' | 1946,2038g/caxis/norm Ocolormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
 % :%s?holdout',1/3,'mcreps',1000?kfold',3,'mcreps',200?gce | %s/1000/200*3/gce
 % :1946,2038g?caxis?norm ocvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
 % :'<,'>s/mean( /cvtcdf(-/gce | '<,'>s/<=/+/gce | '<,'>s/ind) /ind)/gce
 % :exe 'g/mean( 0.5\*(1+sign/s/) );/;/gce' | %s/mean( 0.5\*(1+sign/cvtcdf/gce
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%regf=@(xtrain,ytrain,xtest) polyval(polyfit(xtrain,ytrain,1),xtest);  %TODO no pc
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rng default;
tic;
wrk = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),Hot_n,'kfold',3,'mcreps',200);
ef99arr = wrk;
e99arr = (mean(reshape(wrk,3,[]),1)); e99arr=e99arr(:);
toc; disp(' ');

rng default;
tic;
wrk = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),Hot_n,'holdout',1/3,'mcreps',200*3);
hf99arr = wrk;
h99arr = (mean(reshape(wrk,3,[]),1)); h99arr=h99arr(:);
toc; disp(' ');

%std(reshape(e99arr,[],10)), std(reshape(h99arr,[],10))

rrr.e99 = (squeeze(mean(e99arr,1)));
%np=3;nq=3;
np=6;nq=4;
rng default;
tic;
wrk = crossval(errf,rrr.PERljja_npq(:,np,nq),Hot_n,'kfold',3,'mcreps',200);
e01arr = (mean(reshape(wrk,3,[]),1)); e01arr=e01arr(:);
toc; disp(' ');

rng default;
tic;
wrk = crossval(errf,rrr.PERljja_npq(:,np,nq),Hot_n,'holdout',1/3,'mcreps',200*3);
h01arr = (mean(reshape(wrk,3,[]),1)); h01arr=h01arr(:);
toc; disp(' ');

%std(reshape(e01arr,[],10)), std(reshape(h01arr,[],10))
%std(reshape(e01arr,[],1)) /rrr.e99
%mean(reshape(e01arr,200,[])) /rrr.e99

 % nRegion
 % permute(rrr.e01,[2 3 1])
 % squeeze(permute(rrr.e01(:,3,:,:),[2 3 4 1]))  %D13, D=5

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
%rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)]);
for nr=1:size(mask_xyr,3)
  xdata(:,nr,:,:,:) = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata(:,nr) = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  rng default;
%  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
%  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
%  for np = 1:numel(rrr.r01)
%    if (numel(unique( xdata(:,np) ))>1)
%      rng default;
%      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
%    end
%  end
end % nr nRegion loop
%rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;
corr([xdata(:,2:4,ind(2)) ydata(:,2:4)])
[~,mpq]=min(mean(rrr.e01(3:4,:),1));
mpq=ind(2); wrk=[rrr.e01(2:4,mpq)'; 1-diag(corr([xdata(:,2:4,mpq) ydata(:,2:4)]) ,3)'.^2], wrk(:,1)-mean(wrk(:,2:3),2)
ccc=corr([xdata(:,2:4,mpq) ydata(:,2:4)]); sss=std([xdata(:,2:4,mpq) ydata(:,2:4)]);
% ccc(5,3)=0; ccc(6,2)=0;  ccc(3,2)=0; ccc(2,3)=0;
sss(5:6)*ccc(5:6,2:3)*sss(2:3)'/sqrt(sss(2:3)*ccc(2:3,2:3)*sss(2:3)' * sss(5:6)*ccc(5:6,5:6)*sss(5:6)')


 % produce data, for sigjja vs. e-square, kurtosis
rrr.e99 = (squeeze(mean(e99arr,1)));
e01arr = nan([200 numel(rrr.pArr) numel(rrr.qArr)]);
ef01arr = nan([200*3 numel(rrr.pArr) numel(rrr.qArr)]);
hf01arr = nan([200*3 numel(rrr.pArr) numel(rrr.qArr)]);
for np = 1:numel(rrr.r01)
%  for nq = 1:numel(rrr.qArr)
  rrr.r01( np ) = corr(rrr.PERljja_npq(:, np),Hot_n(:));
  if (numel(unique( rrr.PERljja_npq(:,np) ))>1)
    rng default;
    wrk = crossval(errf,rrr.PERljja_npq(:,np),Hot_n,'kfold',3,'mcreps',200);
    ef01arr(:,np) = wrk;
    e01arr(:,np) = (mean(reshape(wrk,3,[]),1));
    rng default;
    wrk = crossval(errf,rrr.PERljja_npq(:,np),Hot_n,'holdout',1/3,'mcreps',200*3);
    hf01arr(:,np) = wrk;
  end
  rrr.k01( np ) = kurtosis(rrr.PERljja_npq(:, np));
%  end
end
rrr.e01 = (squeeze(mean(e01arr,1)))./rrr.e99;

figure;
sigjja2 = squeeze(mean( 0.5*(1+sign(-e01arr+e99arr))));
xdata=rrr.e01(:); ydata=sigjja2(:);
hold on;
fplot(@(x) tcdf(sqrt(1-x).*sqrt((nyr-2)./x),nyr-2), [0.5 1]);
scatter(xdata,ydata,[],rrr.k01(:));
%scatter(xdata,ydata);
cm=colormap_CD([5/6 0 0.12 1/3 1/2 2/3]',[0.7 .35],[0 0 0 0 0 0],10);
colormap(gca,cm); caxis([2 8]); h=colorbar;
%axis equal; axis([0 0.6 0.4 1.1]);
grid on;
xlabel('e square'); ylabel('sigjja2'); ylabel(h,'Kurtosis');

clf;
sub=rrr.k01(:)<7;
hold on;
fplot(@(x) sqrt(1-x).*sqrt((nyr-2)./x), [0.5 1]);
xdata=rrr.e01(:); ydata=tinv(sigjja2(:),nyr-2); ydata(isinf(ydata))=5;
%scatter(xdata(sub),ydata(sub),[],rrr.k01(sub),'DisplayName','count');
ydata=tinv( tcdf(mean(-e01arr+e99arr)./std(-e01arr+e99arr),length(e99arr)-1) ,nyr-2); ydata=ydata(:);
scatter(xdata(sub),ydata(sub),[],rrr.k01(sub),'^','DisplayName','unsupported t');
%cvtcdf = @(earr) mean( 0.5*(1+sign(earr)) );
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
%ydata=tinv( tcdf(mean(-e01arr+e99arr)./std(-ef01arr+ef99arr)./sqrt(1/length(ef99arr)+0.5),length(ef99arr)-1) ,nyr-2); ydata=ydata(:);
ydata=tinv( cvtcdf(-ef01arr+ef99arr) ,nyr-2); ydata=ydata(:);
scatter(xdata(sub),ydata(sub),[],rrr.k01(sub),'o','DisplayName','BF04');
ydata=tinv( tcdf(mean(-hf01arr+hf99arr)./std(-hf01arr+hf99arr)./sqrt(1/length(hf99arr)+0.5),length(ef99arr)-1) ,nyr-2); ydata=ydata(:);
scatter(xdata(sub),ydata(sub),[],rrr.k01(sub),'+','DisplayName','NB03 resample t');
ydata=tinv( tcdf(mean(-hf01arr+hf99arr)./std(-hf01arr+hf99arr),length(ef99arr)-1) ,nyr-2); ydata=ydata(:);
scatter(xdata(sub),ydata(sub),[],rrr.k01(sub),'v','DisplayName','old');
 % bootstrap r: bootstrap p-value not ready
 % conservative Z: too complicated
 % Markatou et al. 2005 moment: too complicated
legend show;
cm=colormap_CD([5/6 0 0.12 1/3 1/2 2/3]',[0.7 .35],[0 0 0 0 0 0],10);
colormap(gca,cm); caxis([2 8]); h=colorbar;
%axis equal; axis([0 0.6 0.4 1.1]);
grid on;
xlabel('e square'); ylabel('t value'); ylabel(h,'Kurtosis');

np=0;
np=np+1; rrr.D(np),rrr.A(np), clf; hold on; histogram(-e01arr(:,np)+e99arr,12); axis tight; xl=xlim; fplot(@(x) 5*200/12*normpdf( (x-mean(-e01arr(:,np)+e99arr))/std(-e01arr(:,np)+e99arr) ),get(gca,'XLim'))





addpath('/n/home05/pchan/bin');
for np = 1:numel(rrr.pArr)
  for nq = 1:numel(rrr.qArr)
    rrr.k01( np,nq ) = kurtosis(rrr.PERljja_npq(:, np,nq));
  end
end
xdata=rrr.r01(:).^2; ydata=rrr.e01(:);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,1,1); hold on;
%fplot(@(x) 1-x,[0 0.6]);
%fplot(@(x) mean(xdata+ydata)-x,[0 0.6]);
%fplot(@(x) (1-xdata)'*ydata/((1-xdata)'*(1-xdata)) *(1-x),[0 0.6]);
fplot(@(x) (1-xdata(rrr.k01(:)<4))'*ydata(rrr.k01(:)<4)/((1-xdata(rrr.k01(:)<4))'*(1-xdata(rrr.k01(:)<4))) *(1-x),[0 0.6],'m-');
fplot(@(x) (1-xdata(rrr.k01(:)>4&rrr.k01(:)<10))'*ydata(rrr.k01(:)>4&rrr.k01(:)<10)/((1-xdata(rrr.k01(:)>4&rrr.k01(:)<10))'*(1-xdata(rrr.k01(:)>4&rrr.k01(:)<10))) *(1-x),[0 0.6],'b-');
scatter(xdata,ydata,[],rrr.k01(:));
%cm=colormap_CD([0.25 0.90; 0.35 0.7],[.35 .35],[0 0],50);
%cm=flipud(detailCD(10));
cm=colormap_CD([5/6 0 0.12 1/3 1/2 2/3]',[0.7 .35],[0 0 0 0 0 0],10);
%cm=jet(100);
colormap(gca,cm); caxis([2 8]); h=colorbar;
axis equal; axis([0 0.6 0.4 1.1]);
grid on;
xlabel('r square'); ylabel('e square'); ylabel(h,'Kurtosis');
%set(gca,'fontsize',20);
%for np = 1:numel(rrr.pArr)
%  for nq = 1:numel(rrr.qArr)
%    text(double(rrr.r01(np,nq)^2),double(rrr.e01(np,nq)),
%  end
%end
load(['temp_',verX,'_',caseid,'.mat'],'yStart','yEnd');
ydata = Hot_n;
%strYCorr = ['(kurtosis = ',num2str(corr((yStart:yEnd)',ydata(:)),'%+.3f'),')'];
strYCorr = sprintf('kurt=%.1f',kurtosis(ydata));
for m = 1:2
  if (m==1) [~,ind]=min(rrr.e01(:)); end
  if (m==2)
    [~,ind]=max(rrr.r01(:));
    prm.D = rrr.D(ind);
    prm.A = rrr.A(ind);
    strTitle = sprintf('2az6: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
    ver=[strTitle(1:4),'_',caseid];
    PERjja_n=rrr.PERjja_npq(:,ind);
    PERljja_n=rrr.PERljja_npq(:,ind);
    save(['../index_wise/BlockFreq_',verX,ver,'.mat'],'ver','strTitle','prm', 'Hot_n','PERjja_n','PERljja_n','-v7.3');
  end
  xdata = rrr.PERljja_npq(:,ind);
  strT = 'Blocking area on land (km^2)';
  subplot(2,2,m+2); hold on;
  strXCorr = sprintf('kurt=%.1f',kurtosis(xdata));
  ylabel({'Extreme area (km^2)',strYCorr});
  xlabel({strT,strXCorr});
  fplot(@(x) polyval(polyfit(xdata,ydata,1),x), [min(xdata),max(xdata)],'-','linewidth',2);
  for yyyy = yStart:yEnd
    text(double(xdata(yyyy-yStart+1)),double(ydata(yyyy-yStart+1)), sprintf('%02d',mod(yyyy,100)),'HorizontalAlignment','center' );
  end
  title({sprintf('r=%+.3f e^2=%.3f',corr(xdata(:),ydata(:)),rrr.e01(ind))});
  axis square tight; axis([min(xdata),max(xdata), min(ydata),max(ydata)]);
end  % m
fn_figure = ['cv_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

[~,ind]=min(rrr.e01(:));
strTitle = sprintf('2a75: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',rrr.D(ind),rrr.A(ind));
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(:,:)));
%colormap(gca,b2r(-1,1)); colorbar;
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq) = mean( e01arr(:,np,nq)<=e01arr(:,ind) );
      if (sigjja(np,nq)>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(np,nq)),'HorizontalAlignment','center','fontsize',16,'FontWeight','bold');
      else
        text(nq,np,sprintf('%.1f',rrr.e01(np,nq)),'HorizontalAlignment','center','fontsize',16);
%        text(nq,np,replace(sprintf('%.1f',rrr.e01(np,nq)),{'e+0'},{'e'}),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({strTitle},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
print(gcf, '-dpsc2','-append',fn_figure);

[~,ind]=max(rrr.r01(:));
strTitle = sprintf('2a75: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',rrr.D(ind),rrr.A(ind));
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.r01(:,:)));
%colormap(gca,flipud(hot)); colorbar;
colormap(gca,b2r(-.7,.7)); colorbar;
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      text(nq,np,sprintf('%+.2f',rrr.r01(np,nq)),'HorizontalAlignment','center','fontsize',16);
    end
  end
  title({strTitle},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
print(gcf, '-dpsc2','-append',fn_figure);

system(['ps2pdf ',fn_figure]);


rng default;
%cverr = crossval(errf,PERljja_n,Hot_n,'leaveout',1);
%    rng default;
%    wrk(pm,(prma+1)*2) = crossval('mse',rrr.PERljja_npq(:,pm,(prma+1)*2),Hot_n,'Predfun',regf,'mcreps',100); strTitle='10-fold';
%    wrk(pm,(prma+1)*2) = crossval('mse',rrr.PERljja_npq(:,pm,(prma+1)*2),Hot_n,'Predfun',regf,'leaveout',1); strTitle='leave-one-out';
%    wrk(pm,(prma+1)*2) = crossval('mse',rrr.PERljja_npq(:,pm,(prma+1)*2),Hot_n,'Predfun',regf,'holdout',1/3,'mcreps',1000); strTitle='1/3-holdout';
    rng default;
    e01arr(:,pm,(prma+1)*2) = crossval(errf,rrr.PERljja_npq(:,pm,(prma+1)*2),Hot_n,'holdout',1/3,'mcreps',1000); strTitle='1/3-holdout';
%}

%
% tune BlockStat_anomaly_pchan01
 % (18s+31s*6)*9 = 31min
clearvars -except  verX caseid Hot_n Hot_xyn;
prm.D = 1;
prm.R = 0;
try load(['../index_wise/tune_',verX,'2a73_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=round(1-0.5.^([1:7]-1),2);
%rrr.pArr=[0,0.5,0.75,0.88,0.92,0.94,0.95,0.97,0.98];%round(1-0.5.^([1:7]-1),2);
rrr.qArr=[0:0.5:4];
for np = 1:numel(rrr.pArr)
  prm.prst = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.A = rrr.qArr(nq); rrr.nq = nq;
    clearvars -except  verX caseid prm rrr;
    strTitle = sprintf('2a73: DG83, psi500, prst=%g,astd40-90N,S=1,A=%g 15/15day*11yr',prm.prst,prm.A);
    BlockStat_anomaly_pchan01;  % TODO
    rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
    rrr.PERjja_npq(:, rrr.np,rrr.nq) = PERjja_n(:);
    rrr.PERljja_npq(:, rrr.np,rrr.nq) = PERljja_n(:);
    rrr.r00( rrr.np,rrr.nq ) = corr(PERjja_n(:),Hot_n(:));
    rrr.r01( rrr.np,rrr.nq ) = corr(PERljja_n(:),Hot_n(:));
    rrr.r02( rrr.np,rrr.nq ) = corr(PERwjja_n(:),Hotw_n(:));
    rrr.r03( rrr.np,rrr.nq ) = corr(PERlwjja_n(:),Hotw_n(:));
    rrr.prst( rrr.np,rrr.nq ) = prm.prst;
    rrr.A( rrr.np,rrr.nq ) = prm.A;
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr)]);
for nr=1:size(mask_xyr,3)
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'); (sprintf('jja: %s prst=%.2f A=%.1f %+.3g',regArr{nr},rrr.prst(ind(nr)),rrr.A(ind(nr)), rrr.e01(nr,ind(nr)) )),
  %[mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.prst = rrr.prst(ind(nr));
  prm.A = rrr.A(ind(nr));
  titleArr{nr} = sprintf('2a73: DG83, %s prst=%g,A=%g',regArr{nr},prm.prst,prm.A);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('2a73: DG83, psi500, prst=%g,astd40-90N,S=1,A=%g 15/15day*11yr',prm.prst,prm.A);

prm.prst = rrr.prst(ind(1));
prm.A = rrr.A(ind(1));
strTitle = sprintf('2a72: DG83, psi500, prst=%g,astd40-90N,S=1,A=%g 15/15day*11yr',prm.prst,prm.A);
ver=[strTitle(1:4),'_',caseid];
PERjja_n=rrr.PERjja_npq(:,ind(1));
PERljja_n=rrr.PERljja_npq(:,ind(1));
save(['../index_wise/BlockFreq_',verX,ver,'.mat'],'ver','strTitle','prm', 'Hot_n','PERjja_n','PERljja_n','-v7.3');

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq) = cvtcdf(-rrr.e01arr(:,nr,np,nq)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq)>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('A'); ylabel('Persistence factor');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.prst = rrr.prst(ind(2));
prm.A = rrr.A(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('2a73: DG83, psi500, prst=%g,astd40-90N,S=1,A=%g 15/15day*11yr',prm.prst,prm.A);
BlockStat_anomaly_pchan01

 % (20s+36s*6)*9 = 35min
clearvars -except  verX caseid Hot_n Hot_xyn;
prm.D = 1;
prm.R = 1;
try load(['../index_wise/tune_',verX,'2a74_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=[0,0.5,0.75,0.88,0.92,0.94,0.95,0.97,0.98];%round(1-0.5.^([1:7]-1),2);
rrr.qArr=[0:0.5:4];
for np = 1:numel(rrr.pArr)
  prm.prst = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.A = rrr.qArr(nq); rrr.nq = nq;
    clearvars -except  verX caseid prm rrr;
    strTitle = sprintf('2a74: H14, psi500, prst=%g,astd40-90N,S=1,A=%g 15/15day*11yr',prm.prst,prm.A);
    BlockStat_anomaly_pchan01;  % TODO
    rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
    rrr.PERjja_npq(:, rrr.np,rrr.nq) = PERjja_n(:);
    rrr.PERljja_npq(:, rrr.np,rrr.nq) = PERljja_n(:);
    rrr.r00( rrr.np,rrr.nq ) = corr(PERjja_n(:),Hot_n(:));
    rrr.r01( rrr.np,rrr.nq ) = corr(PERljja_n(:),Hot_n(:));
    rrr.r02( rrr.np,rrr.nq ) = corr(PERwjja_n(:),Hotw_n(:));
    rrr.r03( rrr.np,rrr.nq ) = corr(PERlwjja_n(:),Hotw_n(:));
    rrr.prst( rrr.np,rrr.nq ) = prm.prst;
    rrr.A( rrr.np,rrr.nq ) = prm.A;
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr)]);
for nr=1:size(mask_xyr,3)
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'); (sprintf('jja: %s prst=%.2f A=%.1f %+.3g',regArr{nr},rrr.prst(ind(nr)),rrr.A(ind(nr)), rrr.e01(nr,ind(nr)) )),
  %[mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.prst = rrr.prst(ind(nr));
  prm.A = rrr.A(ind(nr));
  titleArr{nr} = sprintf('2a74: H14, %s prst=%g,A=%g',regArr{nr},prm.prst,prm.A);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('2a74: H14, psi500, prst=%g,astd40-90N,S=1,A=%g 15/15day*11yr',prm.prst,prm.A);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq) = cvtcdf(-rrr.e01arr(:,nr,np,nq)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq)>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('A'); ylabel('Persistence factor');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.prst = rrr.prst(ind(2));
prm.A = rrr.A(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('2a74: H14, psi500, prst=%g,astd40-90N,S=1,A=%g 15/15day*11yr',prm.prst,prm.A);
BlockStat_anomaly_pchan01

 % (20s*8)*9 = 24min
clearvars -except  verX caseid Hot_n Hot_xyn;
prm.prst = 0;
prm.R = 0;
try load(['../index_wise/tune_',verX,'2a75_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=[1:2:7, 10:4:22];
rrr.qArr=[0:0.5:4];
for np = 1:numel(rrr.pArr)
  prm.D = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.A = rrr.qArr(nq); rrr.nq = nq;
    clearvars -except  verX caseid prm rrr;
    strTitle = sprintf('2a75: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
    BlockStat_anomaly_pchan01;  % TODO
    rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
    rrr.PERjja_npq(:, rrr.np,rrr.nq) = PERjja_n(:);
    rrr.PERljja_npq(:, rrr.np,rrr.nq) = PERljja_n(:);
    rrr.r00( rrr.np,rrr.nq ) = corr(PERjja_n(:),Hot_n(:));
    rrr.r01( rrr.np,rrr.nq ) = corr(PERljja_n(:),Hot_n(:));
    rrr.r02( rrr.np,rrr.nq ) = corr(PERwjja_n(:),Hotw_n(:));
    rrr.r03( rrr.np,rrr.nq ) = corr(PERlwjja_n(:),Hotw_n(:));
    rrr.D( rrr.np,rrr.nq ) = prm.D;
    rrr.A( rrr.np,rrr.nq ) = prm.A;
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr)]);
for nr=1:size(mask_xyr,3)
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'); (sprintf('jja: %s D=%i A=%.1f %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.A(ind(nr)), rrr.e01(nr,ind(nr)) )),
  %[mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.D = rrr.D(ind(nr));
  prm.A = rrr.A(ind(nr));
  titleArr{nr} = sprintf('2a75: DG83, %s D=%i,A=%g',regArr{nr},prm.D,prm.A);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('2a75: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);

prm.D = rrr.D(ind(1));
prm.A = rrr.A(ind(1));
strTitle = sprintf('2az5: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
ver=[strTitle(1:4),'_',caseid];
PERjja_n=rrr.PERjja_npq(:,ind(1));
PERljja_n=rrr.PERljja_npq(:,ind(1));
save(['../index_wise/BlockFreq_',verX,ver,'.mat'],'ver','strTitle','prm', 'Hot_n','PERjja_n','PERljja_n','-v7.3');

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq) = cvtcdf(-rrr.e01arr(:,nr,np,nq)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq)>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.D = rrr.D(ind(2));
prm.A = rrr.A(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('2a75: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
BlockStat_anomaly_pchan01

 % (20s*8)*9 = 24min
clearvars -except  verX caseid Hot_n Hot_xyn;
prm.prst = 0;
prm.R = 1;
try load(['../index_wise/tune_',verX,'2a76_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=[1:2:7, 10:4:22];
rrr.qArr=[0:0.5:4];
for np = 1:numel(rrr.pArr)
  prm.D = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.A = rrr.qArr(nq); rrr.nq = nq;
    clearvars -except  verX caseid prm rrr;
    strTitle = sprintf('2a76: H14, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
    BlockStat_anomaly_pchan01;  % TODO
    rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
    rrr.PERjja_npq(:, rrr.np,rrr.nq) = PERjja_n(:);
    rrr.PERljja_npq(:, rrr.np,rrr.nq) = PERljja_n(:);
    rrr.r00( rrr.np,rrr.nq ) = corr(PERjja_n(:),Hot_n(:));
    rrr.r01( rrr.np,rrr.nq ) = corr(PERljja_n(:),Hot_n(:));
    rrr.r02( rrr.np,rrr.nq ) = corr(PERwjja_n(:),Hotw_n(:));
    rrr.r03( rrr.np,rrr.nq ) = corr(PERlwjja_n(:),Hotw_n(:));
    rrr.D( rrr.np,rrr.nq ) = prm.D;
    rrr.A( rrr.np,rrr.nq ) = prm.A;
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr)]);
for nr=1:size(mask_xyr,3)
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'); (sprintf('jja: %s D=%i A=%.1f %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.A(ind(nr)), rrr.e01(nr,ind(nr)) )),
  %[mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.D = rrr.D(ind(nr));
  prm.A = rrr.A(ind(nr));
  titleArr{nr} = sprintf('2a76: H14, %s D=%i,A=%g',regArr{nr},prm.D,prm.A);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq) = cvtcdf(-rrr.e01arr(:,nr,np,nq)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq)>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',12);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.D = rrr.D(ind(2));
prm.A = rrr.A(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('2a76: H14, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
BlockStat_anomaly_pchan01
%

% tune BlockStat_d2d_pchan06
 % (38s*8)*5*9 = 228min
clearvars -except  verX caseid Hot_n Hot_xyn;
prm.R = 1;
try load(['../index_wise/tune_',verX,'0a13_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=[1:2:7, 10];
rrr.qArr=[1, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6];
rrr.sArr=[0:0.5:2];
for np = 1:numel(rrr.pArr)
  prm.D = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.S = rrr.qArr(nq); rrr.nq = nq;
    for ns = 1:numel(rrr.sArr)
      prm.A = rrr.sArr(ns); rrr.ns = ns;
      clearvars -except  verX caseid prm rrr;
      strTitle = sprintf('0a13: D13, psi500, D=%i,astd40-90N,S=%g,A=%g 15/15day*11yr',prm.D,prm.S,prm.A);
      BlockStat_d2d_pchan06;  % TODO
      rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq,rrr.ns) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
      rrr.PERjja_npq(:, rrr.np,rrr.nq,rrr.ns) = PERjja_n(:);
      rrr.PERljja_npq(:, rrr.np,rrr.nq,rrr.ns) = PERljja_n(:);
      rrr.r00( rrr.np,rrr.nq,rrr.ns ) = corr(PERjja_n(:),Hot_n(:));
      rrr.r01( rrr.np,rrr.nq,rrr.ns ) = corr(PERljja_n(:),Hot_n(:));
      rrr.r02( rrr.np,rrr.nq,rrr.ns ) = corr(PERwjja_n(:),Hotw_n(:));
      rrr.r03( rrr.np,rrr.nq,rrr.ns ) = corr(PERlwjja_n(:),Hotw_n(:));
      rrr.D( rrr.np,rrr.nq,rrr.ns ) = prm.D;
      rrr.S( rrr.np,rrr.nq,rrr.ns ) = prm.S;
      rrr.A( rrr.np,rrr.nq,rrr.ns ) = prm.A;
    end
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)]);
for nr=1:size(mask_xyr,3)
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'); (sprintf('jja: %s D=%i S=%.2g A=%.1f %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.S(ind(nr)),rrr.A(ind(nr)), rrr.e01(nr,ind(nr)) )),
  [mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.D = rrr.D(ind(nr));
  prm.S = rrr.S(ind(nr));
  prm.A = rrr.A(ind(nr));
  titleArr{nr} = sprintf('0a15: D13, %s D=%i,S=%g,A=%g',regArr{nr},prm.D,prm.S,prm.A);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('0a15: D13, psi500, D=%i,astd40-90N,S=%g,A=%g 15/15day*11yr',prm.D,prm.S,prm.A);

addpath('/n/home05/pchan/bin');
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.sArr),1:numel(rrr.qArr),squeeze(rrr.e01(nr,mp(nr),:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for nq = 1:numel(rrr.qArr)
    for ns = 1:numel(rrr.sArr)
      sigjja(mp(nr),nq,ns) = cvtcdf(-rrr.e01arr(:,nr,mp(nr),nq,ns)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(mp(nr),nq,ns)>=normcdf(-1))
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.qArr)); yticklabels(rrr.qArr);
  xticks(1:numel(rrr.sArr)); xticklabels(rrr.sArr);
xlabel('A'); ylabel('S');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

%{
figure('units','inches','position',[0 1 11 5], 'paperUnits','inches','papersize',[11 5],'paperposition',[0 0 11 5]);
%for nr=1:size(mask_xyr,3)
nr=2;
subplot(1,2,1);
pcolorPH(1:numel(rrr.sArr),1:numel(rrr.qArr),squeeze(rrr.e01(nr,mp(nr),:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); %TODO colorbar;
  for nq = 1:numel(rrr.qArr)
    for ns = 1:numel(rrr.sArr)
%      sigjja(mp(nr),nq,ns) = cvtcdf(-rrr.e01arr(:,nr,mp(nr),nq,ns)+rrr.e01arr(:,nr,ind(nr)));
%      if (sigjja(mp(nr),nq,ns)>=normcdf(-1))
      if (rrr.S(mp(nr),nq,ns)==2.5e6 && rrr.D(mp(nr),nq,ns)==5 && rrr.A(mp(nr),nq,ns)==1.5)
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
%  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.qArr)); yticklabels(rrr.qArr);
  xticks(1:numel(rrr.sArr)); xticklabels(rrr.sArr);
%xlabel('A'); ylabel('S');
axis ij; %axis square;
%set(gca,'FontSize',14);

set(gcf,'units','inches','position',[0 1 11 5], 'paperUnits','inches','papersize',[11 5],'paperposition',[0 0 11 5]);
title(sprintf('D13, when D=%idays',prm.D));
yticks(1:numel(rrr.qArr)); yticklabels([0 rrr.qArr(2:end)]/1e6);
xlabel('A (standard deviations)'); ylabel('S (10^6 km^2)');
%xlabel('Amplitude threshold (standard deviations)'); ylabel('Spatial-scale threshold (10^6 km^2)');
axis equal tight;
set(gca,'FontSize',16);
%end % nr nRegion loop

subplot(1,2,2);
pcolorPH(1:numel(rrr.pArr),1:numel(rrr.qArr),squeeze(rrr.e01(nr,:,:,ms(nr)))'); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
%      sigjja(np,nq,ms(nr)) = cvtcdf(-rrr.e01arr(:,nr,np,nq,ms(nr))+rrr.e01arr(:,nr,ind(nr)));
%      if (sigjja(np,nq,ms(nr))>=normcdf(-1))
%        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16,'Color','m');
%      else
        text(np,nq,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16);
%      end
    end
  end
%  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.pArr)); xticklabels(rrr.pArr);
%xlabel('S'); ylabel('D');
title(sprintf('D13, when A=%.1f standard deviation',prm.A));
yticks(1:numel(rrr.qArr)); yticklabels([0 rrr.qArr(2:end)]/1e6);
xlabel('D (day)'); ylabel('S (10^6 km^2)');
axis ij; %axis square;
axis equal tight;
set(gca,'FontSize',16);

fn_figure = ['corrTable_',verX,'.jpg'];
print(gcf, '-djpeg',fn_figure);
%}

%% save 2 nc for ncl 20190317
cm=colormap_CD([0.45 0.3],[.35 1],[0],50);
ver=['0a13_',caseid];
fn_savenc = ['tune_',verX,ver,'.nc'];
 system(['rm ',fn_savenc]);
nccreate(fn_savenc,'e01','Dimensions',{'reg',size(mask_xyr,3),'np',numel(rrr.pArr),'nq',numel(rrr.qArr),'ns',numel(rrr.sArr)},'DataType','single')  %,'Format','netcdf4'
nccreate(fn_savenc,'cm','Dimensions',{'rgb',3,'ncolors',size(cm,1)},'DataType','single')
nccreate(fn_savenc,'regArr','Dimensions',{'charlen',size(char(regArr)',1),'reg',size(mask_xyr,3)},'DataType','char')
nccreate(fn_savenc,'np','Dimensions',{'np',numel(rrr.pArr)},'DataType','single')
nccreate(fn_savenc,'mp','Dimensions',{'reg',size(mask_xyr,3)},'DataType','int8')
nccreate(fn_savenc,'nq','Dimensions',{'nq',numel(rrr.qArr)},'DataType','single')
nccreate(fn_savenc,'mq','Dimensions',{'reg',size(mask_xyr,3)},'DataType','int8')
nccreate(fn_savenc,'ns','Dimensions',{'ns',numel(rrr.sArr)},'DataType','single')
nccreate(fn_savenc,'ms','Dimensions',{'reg',size(mask_xyr,3)},'DataType','int8')
ncwriteatt(fn_savenc,'/','description','D13')
ncwriteatt(fn_savenc,'np','description','D (day)')
ncwriteatt(fn_savenc,'nq','description','S (10~S~6~N~ km~S~2~N~)')
ncwriteatt(fn_savenc,'ns','description','A (standard deviations)')
ncwrite(fn_savenc,'e01',rrr.e01)
ncwrite(fn_savenc,'cm',cm')
ncwrite(fn_savenc,'regArr',char(regArr)')
ncwrite(fn_savenc,'np',rrr.pArr)
ncwrite(fn_savenc,'mp',mp)
ncwrite(fn_savenc,'nq',[0 rrr.qArr(2:end)]/1e6)  %rrr.qArr)
ncwrite(fn_savenc,'mq',mq)
ncwrite(fn_savenc,'ns',rrr.sArr)
ncwrite(fn_savenc,'ms',ms)
%system(['ln -sf ',fn_savenc,' pointcorr.nc']);
%system(['ncl -Q xy.tune_param.ncl']);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.sArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,mq(nr),:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for ns = 1:numel(rrr.sArr)
      sigjja(np,mq(nr),ns) = cvtcdf(-rrr.e01arr(:,nr,np,mq(nr),ns)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,mq(nr),ns)>=normcdf(-1))
        text(ns,np,sprintf('%.2f',rrr.e01(nr,np,mq(nr),ns)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(ns,np,sprintf('%.2f',rrr.e01(nr,np,mq(nr),ns)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.sArr)); xticklabels(rrr.sArr);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:,ms(nr)))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq,ms(nr)) = cvtcdf(-rrr.e01arr(:,nr,np,nq,ms(nr))+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq,ms(nr))>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('S'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.D = rrr.D(ind(2));
prm.S = rrr.S(ind(2));
prm.A = rrr.A(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('0a15: D13, psi500, D=%i,astd40-90N,S=%g,A=%g 15/15day*11yr',prm.D,prm.S,prm.A);
BlockStat_d2d_pchan06
%

% tune BlockStat_scherrer_pchan05
 % (13s*7)*7*5 = 53min
clearvars -except  verX caseid Hot_n Hot_xyn;
prm.GHGST = 0;
try load(['../index_wise/tune_',verX,'4a26_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=[1:2:7];
rrr.qArr=[3:9];
rrr.sArr=[-20:5:0, Inf];
for np = 1:numel(rrr.pArr)
  prm.D = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.dphi = rrr.qArr(nq); rrr.nq = nq;
    for ns = 1:numel(rrr.sArr)
      prm.GHGNT = rrr.sArr(ns); rrr.ns = ns;
      clearvars -except  verX caseid prm rrr;
      strTitle = sprintf('4a26: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);
      BlockStat_scherrer_pchan05;  % TODO
      rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq,rrr.ns) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
      rrr.PERjja_npq(:, rrr.np,rrr.nq,rrr.ns) = PERjja_n(:);
      rrr.PERljja_npq(:, rrr.np,rrr.nq,rrr.ns) = PERljja_n(:);
      rrr.r00( rrr.np,rrr.nq,rrr.ns ) = corr(PERjja_n(:),Hot_n(:));
      rrr.r01( rrr.np,rrr.nq,rrr.ns ) = corr(PERljja_n(:),Hot_n(:));
      rrr.r02( rrr.np,rrr.nq,rrr.ns ) = corr(PERwjja_n(:),Hotw_n(:));
      rrr.r03( rrr.np,rrr.nq,rrr.ns ) = corr(PERlwjja_n(:),Hotw_n(:));
      rrr.D( rrr.np,rrr.nq,rrr.ns ) = prm.D;
      rrr.dphi( rrr.np,rrr.nq,rrr.ns ) = prm.dphi;
      rrr.GHGNT( rrr.np,rrr.nq,rrr.ns ) = prm.GHGNT;
    end
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

%rrr=rmfield(rrr,{'e01','e99'});
load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)]);
for nr=1:size(mask_xyr,3)
%  xdata = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*mask_xyr(:,:,nr),[ds(1:2),nd_jja,nyr]),3),1))]';
%  xdata = [areaEarth * squeeze(mean(rbig.PERjja_xynpq.*mask_xyr(:,:,nr),1))]';
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
%  rrr.e01(nr,:) = (squeeze(mean(rrr.e01arr,1)))./rrr.e99(nr);
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'+99*(isinf(rrr.GHGNT(:)))); (sprintf('jja: %s D=%i dphi=%i GHGNT=%i %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.dphi(ind(nr)),rrr.GHGNT(ind(nr)), rrr.e01(nr,ind(nr)) )),
  [mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.D = rrr.D(ind(nr));
  prm.dphi = rrr.dphi(ind(nr));
  prm.GHGNT = rrr.GHGNT(ind(nr));
  titleArr{nr} = sprintf('4a26: S06, %s D=%i,GHGNT=%g,dphi=%i',regArr{nr},prm.D,prm.GHGNT,prm.dphi);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%  strTitle = sprintf('4a26: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);

addpath('/n/home05/pchan/bin');
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.sArr),1:numel(rrr.qArr),squeeze(rrr.e01(nr,mp(nr),:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for nq = 1:numel(rrr.qArr)
    for ns = 1:numel(rrr.sArr)
      sigjja(mp(nr),nq,ns) = cvtcdf(-rrr.e01arr(:,nr,mp(nr),nq,ns)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(mp(nr),nq,ns)>=normcdf(-1))
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.qArr)); yticklabels(rrr.qArr);
  xticks(1:numel(rrr.sArr)); xticklabels(rrr.sArr);
xlabel('GHGNT'); ylabel('dphi');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.sArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,mq(nr),:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for ns = 1:numel(rrr.sArr)
      sigjja(np,mq(nr),ns) = cvtcdf(-rrr.e01arr(:,nr,np,mq(nr),ns)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,mq(nr),ns)>=normcdf(-1))
        text(ns,np,sprintf('%.2f',rrr.e01(nr,np,mq(nr),ns)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(ns,np,sprintf('%.2f',rrr.e01(nr,np,mq(nr),ns)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.sArr)); xticklabels(rrr.sArr);
xlabel('GHGNT'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:,ms(nr)))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq,ms(nr)) = cvtcdf(-rrr.e01arr(:,nr,np,nq,ms(nr))+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq,ms(nr))>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('dphi'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.D = rrr.D(ind(2));
prm.dphi = rrr.dphi(ind(2));
prm.GHGNT = rrr.GHGNT(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('4a29: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);
BlockStat_scherrer_pchan05


clearvars -except  verX caseid Hot_n Hot_xyn;
prm.GHGST = 0;
load(['../index_wise/tune_',verX,'4a26_',caseid,'.mat'],'rrr','rbig');

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'+99*(~isinf(rrr.GHGNT(:)))); (sprintf('jja: %s D=%i dphi=%i GHGNT=%i %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.dphi(ind(nr)),rrr.GHGNT(ind(nr)), rrr.e01(nr,ind(nr)) )),
  [mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.D = rrr.D(ind(nr));
  prm.dphi = rrr.dphi(ind(nr));
  prm.GHGNT = rrr.GHGNT(ind(nr));
  titleArr{nr} = sprintf('4a39: S06, %s D=%i,GHGNT=%g,dphi=%i',regArr{nr},prm.D,prm.GHGNT,prm.dphi);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('4a39: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);

addpath('/n/home05/pchan/bin');
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:,ms(nr)))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq,ms(nr)) = cvtcdf(-rrr.e01arr(:,nr,np,nq,ms(nr))+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq,ms(nr))>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('dphi'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.D = rrr.D(ind(2));
prm.dphi = rrr.dphi(ind(2));
prm.GHGNT = rrr.GHGNT(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('4a39: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);
BlockStat_scherrer_pchan05
%disp('return'); return;

% tune BlockStat_m2d_pchan02
 % (14s+30..200s*4)*5 = 40min
%
clearvars -except  verX caseid Hot_n Hot_xyn;
try load(['../index_wise/tune_',verX,'8a14_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=[1,2,3,5,7];
rrr.qArr=[1:2:9];
for np = 1:numel(rrr.pArr)
  prm.D = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.dphi = rrr.qArr(nq); rrr.nq = nq;
    clearvars -except  verX caseid prm rrr;
    strTitle = sprintf('8a14: M13, Z500, 40-74N, D=%i,dphi=%i,smth',prm.D,prm.dphi);
    BlockStat_m2d_pchan02;  % TODO
    rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
    rrr.PERjja_npq(:, rrr.np,rrr.nq) = PERjja_n(:);
    rrr.PERljja_npq(:, rrr.np,rrr.nq) = PERljja_n(:);
    rrr.r00( rrr.np,rrr.nq ) = corr(PERjja_n(:),Hot_n(:));
    rrr.r01( rrr.np,rrr.nq ) = corr(PERljja_n(:),Hot_n(:));
    rrr.r02( rrr.np,rrr.nq ) = corr(PERwjja_n(:),Hotw_n(:));
    rrr.r03( rrr.np,rrr.nq ) = corr(PERlwjja_n(:),Hotw_n(:));
    rrr.D( rrr.np,rrr.nq ) = prm.D;
    rrr.dphi( rrr.np,rrr.nq ) = prm.dphi;
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr)]);
for nr=1:size(mask_xyr,3)
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'); (sprintf('jja: %s D=%i dphi=%i %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.dphi(ind(nr)), rrr.e01(nr,ind(nr)) )),
  %[mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.D = rrr.D(ind(nr));
  prm.dphi = rrr.dphi(ind(nr));
  titleArr{nr} = sprintf('8a15: M13, %s D=%i,dphi=%i',regArr{nr},prm.D,prm.dphi);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('8a15: M13, Z500, 40-74N, D=%i,dphi=%i,smth',prm.D,prm.dphi);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq) = cvtcdf(-rrr.e01arr(:,nr,np,nq)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq)>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('dphi'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.D = rrr.D(ind(2));
prm.dphi = rrr.dphi(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('8a15: M13, Z500, 40-74N, D=%i,dphi=%i,smth',prm.D,prm.dphi);
BlockStat_m2d_pchan02

% tune BlockStat_falwa_pchan01
 % (19s+110..240s*4)*6*6 = 431min
clearvars -except  verX caseid Hot_n Hot_xyn;
try load(['../index_wise/tune_',verX,'9a06_',caseid,'.mat'],'rrr','rbig'); end
if (exist('rrr','var')==0)
rrr.pArr=[1,2,3,5,7];
rrr.qArr=[-6:-1];
rrr.sArr=[0,2e7,4e7,6e7,7.2e7,8e7];
for np = 1:numel(rrr.pArr)
  prm.D = rrr.pArr(np); rrr.np = np;
  for nq = 1:numel(rrr.qArr)
    prm.joffset = rrr.qArr(nq); rrr.nq = nq;
    for ns = 1:numel(rrr.sArr)
      prm.A = rrr.sArr(ns); rrr.ns = ns;
      clearvars -except  verX caseid prm rrr;
      strTitle = sprintf('9a06: FALWA, z500, A+,40-90N,offset%i,A=%.1g,D=%i,unweight',prm.joffset,prm.A,prm.D);
      BlockStat_falwa_pchan01;  % TODO
      rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq,rrr.ns) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
      rrr.PERjja_npq(:, rrr.np,rrr.nq,rrr.ns) = PERjja_n(:);
      rrr.PERljja_npq(:, rrr.np,rrr.nq,rrr.ns) = PERljja_n(:);
      rrr.r00( rrr.np,rrr.nq,rrr.ns ) = corr(PERjja_n(:),Hot_n(:));
      rrr.r01( rrr.np,rrr.nq,rrr.ns ) = corr(PERljja_n(:),Hot_n(:));
      rrr.r02( rrr.np,rrr.nq,rrr.ns ) = corr(PERwjja_n(:),Hotw_n(:));
      rrr.r03( rrr.np,rrr.nq,rrr.ns ) = corr(PERlwjja_n(:),Hotw_n(:));
      rrr.D( rrr.np,rrr.nq,rrr.ns ) = prm.D;
      rrr.joffset( rrr.np,rrr.nq,rrr.ns ) = prm.joffset;
      rrr.A( rrr.np,rrr.nq,rrr.ns ) = prm.A;
    end
  end
end
rbig.PERjja_xynpq = rrr.PERjja_xynpq; rrr=rmfield(rrr,'PERjja_xynpq');
end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'areaEarth','mask_xyr','regArr');
regf=@(xtrain,ytrain,xtest) polyval(polyfitPC(xtrain,ytrain,1),xtest);
%errf=@(xtrain,ytrain,xtest,ytest) ytest-regf(xtrain,ytrain,xtest);
errf=@(xtrain,ytrain,xtest,ytest) (mean((ytest-regf(xtrain,ytrain,xtest)).^2));
rrr.e01arr = nan([200*3 size(mask_xyr,3) numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)]);
for nr=1:size(mask_xyr,3)
  xdata = squeeze(mean(sum( areaEarth.*rbig.PERjja_xynpq.*mask_xyr(:,:,nr) ,2),1));
  ydata = squeeze(mean(sum( areaEarth.*Hot_xyn.*mask_xyr(:,:,nr) ,2),1));
%  [~,nmax]=max(ydata); ydata(nmax)=[]; xdata(nmax,:)=[]; %TODO
  rng default;
  e99arr = crossval(@(ytrain,ytest) (mean((ytest-mean(ytrain)).^2)),ydata,'kfold',3,'mcreps',200);
  rrr.e99(nr,1) = (squeeze(mean(e99arr,1)));
  for np = 1:numel(rrr.r01)
    if (numel(unique( xdata(:,np) ))>1)
      rng default;
      rrr.e01arr(:,nr,np) = crossval(errf,xdata(:,np),ydata,'kfold',3,'mcreps',200);
    end
  end
end % nr nRegion loop
rrr.e01 = (squeeze(mean(rrr.e01arr,1)))./rrr.e99;

if (exist('strTitle','var'))
fn_save  = ['../index_wise/tune_',verX,ver,'.mat'];
save(fn_save,'prm','rrr','rbig','strTitle');
end  % exist strTitle
%TODO end  % exist rrr

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'); (sprintf('jja: %s D=%i joffset=%i A=%.2g %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.joffset(ind(nr)),rrr.A(ind(nr)), rrr.e01(nr,ind(nr)) )),
  [mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
  prm.D = rrr.D(ind(nr));
  prm.joffset = rrr.joffset(ind(nr));
  prm.A = rrr.A(ind(nr));
  titleArr{nr} = sprintf('9a07: FALWA, %s offset%i,A=%.1g,D=%i',regArr{nr},prm.joffset,prm.A,prm.D);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('9a07: FALWA, z500, A+,40-90N,offset%i,A=%.1g,D=%i,unweight',prm.joffset,prm.A,prm.D);

addpath('/n/home05/pchan/bin');
cvtcdf = @(earr) tcdf( mean(earr)./max(std(earr),std(Hot_n)/1e13)./sqrt(1/length(earr)+0.5), length(earr)-1);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.sArr),1:numel(rrr.qArr),squeeze(rrr.e01(nr,mp(nr),:,:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for nq = 1:numel(rrr.qArr)
    for ns = 1:numel(rrr.sArr)
      sigjja(mp(nr),nq,ns) = cvtcdf(-rrr.e01arr(:,nr,mp(nr),nq,ns)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(mp(nr),nq,ns)>=normcdf(-1))
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(ns,nq,sprintf('%.2f',rrr.e01(nr,mp(nr),nq,ns)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.qArr)); yticklabels(rrr.qArr);
  xticks(1:numel(rrr.sArr)); xticklabels(rrr.sArr);
xlabel('A'); ylabel('joffset');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.sArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,mq(nr),:))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for ns = 1:numel(rrr.sArr)
      sigjja(np,mq(nr),ns) = cvtcdf(-rrr.e01arr(:,nr,np,mq(nr),ns)+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,mq(nr),ns)>=normcdf(-1))
        text(ns,np,sprintf('%.2f',rrr.e01(nr,np,mq(nr),ns)),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(ns,np,sprintf('%.2f',rrr.e01(nr,np,mq(nr),ns)),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.sArr)); xticklabels(rrr.sArr);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
for nr=1:size(mask_xyr,3)
subplot(2,2,nr);
pcolorPH(1:numel(rrr.qArr),1:numel(rrr.pArr),squeeze(rrr.e01(nr,:,:,ms(nr)))); shading faceted;
%colormap(gca,b2r(-1,1)); colorbar;
colormap(gca,colormap_CD([0.45 0.3],[.35 1],[0],50));
caxis([0.5 1]); colorbar;
  for np = 1:numel(rrr.pArr)
    for nq = 1:numel(rrr.qArr)
      sigjja(np,nq,ms(nr)) = cvtcdf(-rrr.e01arr(:,nr,np,nq,ms(nr))+rrr.e01arr(:,nr,ind(nr)));
      if (sigjja(np,nq,ms(nr))>=normcdf(-1))
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16,'Color','m');
      else
        text(nq,np,sprintf('%.2f',rrr.e01(nr,np,nq,ms(nr))),'HorizontalAlignment','center','fontsize',16);
      end
    end
  end
  title({titleArr{nr}},'FontSize',20); yticks(1:numel(rrr.pArr)); yticklabels(rrr.pArr);
  xticks(1:numel(rrr.qArr)); xticklabels(rrr.qArr);
xlabel('joffset'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
end % nr nRegion loop
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);

prm.D = rrr.D(ind(2));
prm.joffset = rrr.joffset(ind(2));
prm.A = rrr.A(ind(2));
clearvars -except  verX caseid prm;
strTitle = sprintf('9a07: FALWA, z500, A+,40-90N,offset%i,A=%.1g,D=%i,unweight',prm.joffset,prm.A,prm.D);
BlockStat_falwa_pchan01
%

%
fn_figure = ['corrTable_',verX,'.ps'];
system(['ps2pdf ',fn_figure]);
%

%x = fminsearch(@rneg_scherrer,[104 100-10 100 53],optimset('TolFun',2e-3,'TolX',1e-3))
%clear prm
%prm.D = 1+abs(round(x(1)-100));
%prm.GHGNT = x(2)-100;
%prm.GHGST = x(3)-100;
%prm.dphi = 1+abs(-10+mod(10+round(x(4)-50),20));
%clearvars -except  verX caseid prm;
%strTitle = sprintf('4a28: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);
%BlockStat_scherrer_pchan05;

%clearvars -except  verX caseid;
%prm.R = 0;
%prm.D = 5;
%prm.A = 1.5;
%prm.prst = 0;
%strTitle = sprintf('2a68: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
%BlockStat_anomaly_pchan01

clearvars -except  verX caseid;
prm.R = 0;
prm.D = 1;
prm.A = 0;
prm.prst = 0;
strTitle = sprintf('2a69: DG83, psi500, D=%i,astd40-90N,S=1,A=%g 15/15day*11yr',prm.D,prm.A);
BlockStat_anomaly_pchan01





clearvars -except  verX caseid;
prm.dphi = 5;
prm.GHGST = 0;
prm.GHGNT = -10.;
prm.D = 5;
%strTitle = sprintf('4a26: S06, 40-74N, D=%i,GHGNT=%g',prm.D,prm.GHGNT);
strTitle = sprintf('4a26: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);
BlockStat_scherrer_pchan05

clearvars -except  verX caseid;
prm.dphi = 5;
prm.D = 5;
strTitle = sprintf('8a14: M13, Z500, 40-74N, D=%i,dphi=%i,smth',prm.D,prm.dphi);
BlockStat_m2d_pchan02

%disp('return'); return;
clearvars -except  verX caseid;
load(['../index_wise/tune_',verX,'9a06_',caseid,'.mat'],'rrr');
%[rmax,ind] = max(rrr.r01(:)-99*(rrr.D(:)~=5 | rrr.A(:)~=7.2e7));

load(['temp_',verX,'_',caseid,'.mat'],'mask_xyr','regArr');
for nr=1:size(mask_xyr,3)
  [emin(nr),ind(nr)] = min(rrr.e01(nr,:)'+99*(rrr.D(:)~=5 | rrr.A(:)~=7.2e7)); (sprintf('jja: %s D=%i joffset=%i A=%.2g %+.3g',regArr{nr},rrr.D(ind(nr)),rrr.joffset(ind(nr)),rrr.A(ind(nr)), rrr.e01(nr,ind(nr)) )),
%  [mp(nr),mq(nr),ms(nr)] = ind2sub([numel(rrr.pArr) numel(rrr.qArr) numel(rrr.sArr)], ind(nr));
%  prm.D = rrr.D(ind(nr));
%  prm.joffset = rrr.joffset(ind(nr));
%  prm.A = rrr.A(ind(nr));
%  titleArr{nr} = sprintf('9a07: FALWA, %s offset%i,A=%.1g,D=%i',regArr{nr},prm.joffset,prm.A,prm.D);
end % nr nRegion loop
if (emin(2)>=emin(1)) disp(sprintf('mine01=%+.3g > mine00=%+.3g',emin(2),emin(1))); end
%strTitle = sprintf('9a07: FALWA, z500, A+,40-90N,offset%i,A=%.1g,D=%i,unweight',prm.joffset,prm.A,prm.D);

prm.D = rrr.D(ind(2));
prm.joffset = rrr.joffset(ind(2));
prm.A = rrr.A(ind(2));
clearvars -except  verX caseid prm;
%prm.joffset = -3;
%prm.D = 5;
%prm.A = -1;
strTitle = sprintf('9a06: FALWA, z500, A+,40-90N,offset%i,A=%.1g,D=%i,unweight',prm.joffset,prm.A,prm.D);
BlockStat_falwa_pchan01
%
!mv all*.pdf ../index_wise/

%%
%cd ../matlab
addpath('/n/home05/pchan/bin');
%  verX='x919';
%  caseid=['ERA-interim_19790101-20171231'];
clearvars -except  verX caseid;
%miscArr = {'2a73','DG83,prst=.5,A=1',50.; '2a75','DG83,D=5,A=.5',75.; '2a68','DG83,D=5,A=1.5',50; '0a15','D13,D=1,S=1,A=1',75.; '0a13','D13,D=5,S=2.5e6,A=1.5',20; '4a29','S06,D=1,GHGN=-15,GHGS=0,dphi=7',75.; '4a26','S06,D=5,GHGN=-10,GHGS=0,dphi=5',50; '8a15','M13,D=1,dphi=9',50; '8a14','M13,D=5,dphi=5',50; '9a07','FALWA,D=3,A=4e7,offset-3',75.; '9a06','FALWA,D=5,A=median,offset-3',75; verX,'Extreme',50.};  % 180827
 %  '4a28','S06,D=5,GHGN=-3,GHGS=-5,dphi=7',100.;
%miscArr = {'2a73','DG83,prst=.94,A=2.5',50.; '2a75','DG83,D=3,A=1',75.; '2a68','DG83,D=5,A=1.5',50; '0a15','D13,D=1,S=1,A=1',75.; '0a13','D13,D=5,S=2.5e6,A=1.5',20; '4a28','S06,D=6,GHGN=-2,GHGS=-5,dphi=7',100; '4a29','S06,D=1,GHGN=-15,GHGS=0,dphi=7',75.; '4a26','S06,D=5,GHGN=-10,GHGS=0,dphi=5',50; '8a14','M13,D=5,dphi=5',50; '9a07','FALWA,D=3,A=4e7,offset-2',75.; '9a06','FALWA,D=5,A=median,offset-3',75; verX,'Extreme',50.};  % 180827
 %  '2a69','DG83,D=1,A=0',100;
%miscArr = {'2a73','',50.; '2a75','',75.; '2a68','',50; '0a15','',75.; '0a13','',20; '4a29','',75.; '4a26','',50; '8a15','',50; '8a14','',50; '9a07','',75.; '9a06','',75; verX,'',50.};  % 180827
%miscArr = {'2a73','',50.; '2a75','',75.; '0a15','',75.; '0a13','',20; '4a29','',75.; '4a26','',50; '4a39','',75.; '8a15','',50; '8a14','',50; '9a07','',75.; '9a06','',75; verX,'',50.};  % 181014
%miscArr = {'2a73','tuned DG83p',50.; '2a75','tuned DG83',75.; '0a15','tuned D13',75.; '0a13','D13',20; '4a29','tuned S06',75.; '4a26','S06',50; '8a14','M13',50; '9a07','tuned M17',75.; '9a06','M17',75; verX,'Extreme',50.};  % 181014
miscArr = {'2a73','tuned DG83p',50.; '2a75','tuned DG83',75.; '0a15','tuned D13',75.; '0a13','D13',20; '4a29','tuned S06',75.; '4a26','S06',50; '8a15','tuned M13',50.; '8a14','M13',50; '9a07','tuned M17',75.; '9a06','M17',75; verX,'Extreme',50.};  % 181124
vernumArr = miscArr(:,1);
titleArr = miscArr(:,2);  %TODO vernumArr;
lims = cell2mat(miscArr(:,3));
%vernumArr = {'2a73','2a75','2a68','2a69','0a15','0a13','4a28','4a26','8a14','9a07','9a06',verX};  % 180809
%titleArr = {'DG83,prst=.5,A=1','DG83,D=5,A=1.5','DG83,D=1,A=0','D13,D=5,S=2.5e6,A=1.5','S06,D=5,GHGN=-10','M13,D=5','FALWA,D=5,A=median','Extreme'};
xtrm_reanalysis_pchan04


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
%{

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
  xlabel({'Extreme area (km^2)',strXCorr});
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
  xlabel({'Extreme area (km^2)',strXCorr});
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
%}

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
  xlabel({'day'}); ylabel({'Weighted extreme area (unit*km^2)'});
else
  xlabel({'day'}); ylabel({'Extreme area (km^2)'});
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
  xlabel({'year'}); ylabel({'Mean weighted extreme area (unit*km^2)'});
else
  xlabel({'year'}); ylabel({'Mean extreme area (km^2)'});
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
  xlabel({'day'}); ylabel({'Weighted extreme area (unit*km^2)'});
else
  xlabel({'day'}); ylabel({'Extreme area (km^2)'});
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
  xlabel({'year'}); ylabel({'Mean weighted extreme area (unit*km^2)'});
else
  xlabel({'year'}); ylabel({'Mean extreme area (km^2)'});
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
PERfreq_jja(PERfreq_jja==0) = nan;
PERfreq_djf(PERfreq_djf==0) = nan;
%PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
%PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_jja'); shading flat;
colormap(gca,jet(12)); caxis([0 12]); colorbar; caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
%title('\fontsize{20}Relative frequency (%) of intense extreme events during JJA');
title({strTitle,'JJA extreme frequency (%)'},'fontsize',16);
tightmap;
%print(gcf,'-dpdf',['Pfahl2a_',textTH,'_',text,'.pdf'])
%savefig(gcf,['Pfahl2a_',textTH,'_',text,'.fig'])

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(gca,jet(12)); caxis([0 12]); colorbar; caxis auto;
plotm(coastlat,coastlon,'k')
%title('\fontsize{20}Relative frequency (%) of intense extreme events during DJF');
title({'DJF extreme frequency (%)'},'fontsize',16);
tightmap;
%pause(5);
%print(gcf,'-dpdf',['Pfahl2c_',textTH,'_',text,'.pdf'])
%savefig(gcf,['Pfahl2c_',textTH,'_',text,'.fig'])
print(gcf, '-dpsc2','-append',fn_figure);
%

% wgt
PERfreq_jja = mean(PER_jja.*(Wgt_jja),3);
PERfreq_djf = mean(PER_djf.*(Wgt_djf),3);
PERfreq_jja(PERfreq_jja==0) = nan;
PERfreq_djf(PERfreq_djf==0) = nan;
%PERfreq_jja(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;  % TODO
%PERfreq_djf(:,[1:prm.yN1-1,prm.yN2+1:end]) = nan;

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_jja'); shading flat;
colormap(gca,jet(12)); caxis([0 12]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k');
title({strTitle,'JJA weighted extreme frequency (unit*%)'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(gca,jet(12)); caxis([0 24]); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k');
title({'DJF weighted extreme frequency (unit*%)'},'fontsize',16);
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);

%% POD (Pfahl2b in xtrm_colocate_pchan)
%{
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolorm(double(lat1a),double(lon1a),double(100*HotPod5000'));
colormap(gca,jet(10)); caxis([0 100]); colorbar;
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

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERjja_reg'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-5e-6,5e-6)); %colorbar;  TODO
  title({strTitle,['JJA extreme regressed on extreme hot area'],'unit*%/K/km^2'},'fontsize',16);
else
  colormap(gca,b2r(-5e-6,5e-6)); %colorbar;
  title({strTitle,['JJA extreme regressed on extreme hot area'],'%/km^2'},'fontsize',16);
end
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERdjf_reg'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-5e-6,5e-6)); %colorbar;
  title({['DJF extreme regressed on extreme cold area'],'unit*%/K/km^2'},'fontsize',16);
else
  colormap(gca,b2r(-5e-6,5e-6)); %colorbar;
  title({['DJF extreme regressed on extreme cold area'],'%/km^2'},'fontsize',16);
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

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERjja_trend'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-1,1)); colorbar; %caxis auto;  %TODO
  title({strTitle,['JJA weighted extreme frequency trends'],'unit*%/yr'},'fontsize',16);
else
  colormap(gca,b2r(-0.15,0.15)); colorbar; %caxis auto;
  title({strTitle,['JJA extreme frequency trends'],'%/yr'},'fontsize',16);
end
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERdjf_trend'); shading flat;
plotm(coastlat,coastlon,'k')
if (contains(textTH, 'wgt'))
  colormap(gca,b2r(-1.5,1.5)); colorbar; %caxis auto;  %TODO
  title({['DJF weighted extreme frequency trends'],'unit*%/yr'},'fontsize',16);
else
  colormap(gca,b2r(-0.15,0.15)); colorbar; %caxis auto;
  title({['DJF extreme frequency trends'],'%/yr'},'fontsize',16);
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
%title({['JJA extreme singular vector ',num2str(m)]},'fontsize',16);
title({['JJA extreme #',num2str(m)]},'fontsize',16);
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
%  ylabel({'Mean weighted extreme area (unit*km^2)'});
%else
%  ylabel({'Mean extreme area (km^2)'});
%end

end %m loop
if (contains(textTH, 'wgt'))
  subplot(3,4,9); yyaxis left; ylabel({'Mean weighted extreme area (K*km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean weighted extreme area (unit*km^2)'});
else
  subplot(3,4,9); yyaxis left; ylabel({'Mean extreme area (km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean extreme area (km^2)'});
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
%title({['DJF extreme singular vector ',num2str(m)]},'fontsize',16);
title({['DJF extreme #',num2str(m)]},'fontsize',16);
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
%  ylabel({'Mean weighted extreme area (unit*km^2)'});
%else
%  ylabel({'Mean extreme area (km^2)'});
%end

end %m loop
if (contains(textTH, 'wgt'))
  subplot(3,4,9); yyaxis left; ylabel({'Mean weighted extreme area (K*km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean weighted extreme area (unit*km^2)'});
else
  subplot(3,4,9); yyaxis left; ylabel({'Mean extreme area (km^2)'});
  subplot(3,4,12); yyaxis right; ylabel({'Mean extreme area (km^2)'});
end
print(gcf, '-dpsc2','-append',fn_figure);

end  % textWgt
%}


system(['ps2pdf ',fn_figure]);
system(['rm ',fn_figure]);
toc
%}


%% plot quantile (cf xtrmfreq)
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh' ,'latt42','lont42' ,'lsm_jja','HotQuantile' ,'lsm_djf','ColdQuantile')

addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(2,1,1);
%axesm('MapProjection','ortho','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20)
%axesm('MapProjection','vperspec','origin',[90 0],'MapParallels',6000,'grid','on','mlinelocation',20,'plinelocation',20);
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
HotQuantile(~lsm_jja) = nan;
%[cc,hc]=contourfm(lat1a,lon1a,double(HotQuantile'));
%pcolorm(double(latt42),double(lont42([end/2:end,1:end/2])),double(HotQuantile([end/2:end,1:end/2],:)')); shading flat;  % cyclic point added
pcolormPC(latt42,lont42,HotQuantile'); shading flat;
%contourm(latt42,lonplot,double(HotQuantile([1:end,1],:))',[1:12]); axis equal tight;
%clabelm(cc,hc,[-2:1:1,1.2])%,'BackgroundColor','none')
%colormap(gca,jet(10)); caxis([0 40]); colorbar;  % old: all season
colormap(gca,jet(12)); caxis([0 12]); colorbar;
plotm(coastlat,coastlon,'k')
%title('\fontsize{20}The 99th percentile of six-hourly maximum temperature T_{max}');
%title({'99th percentile of daily T_{max}','(all season mean removed)'},'fontsize',16);  % old: all season
title({'99th percentile of daily T_{max}','(JJA mean removed)'},'fontsize',16);
xlim([-pi pi]); ylim([0 pi/2]);  % tightmap??

subplot(2,1,2);
%axesm('MapProjection','ortho','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20)
%axesm('MapProjection','vperspec','origin',[90 0],'MapParallels',6000,'grid','on','mlinelocation',20,'plinelocation',20);
axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
ColdQuantile(~lsm_djf) = nan;
%[cc,hc]=contourfm(lat1a,lon1a,double(-ColdQuantile'));
%pcolorm(double(latt42),double(lont42([end/2:end,1:end/2])),double(-ColdQuantile([end/2:end,1:end/2],:)')); shading flat;  % cyclic point added
pcolormPC(latt42,lont42,-ColdQuantile'); shading flat;
%contourm(latt42,lonplot,double(-ColdQuantile([1:end,1],:))',[-24:2:-2]); axis equal tight;
%clabelm(cc,hc,[-2:1:1,1.2])%,'BackgroundColor','none')
%colormap(gca,jet(10)); caxis([-40 0]); colorbar;  % old: all season
colormap(gca,jet(12)); caxis([-24 0]); colorbar;
plotm(coastlat,coastlon,'k')
%title({'1st percentile of daily T_{min}','(all season mean removed)'},'fontsize',16);  % old: all season
title({'1st percentile of daily T_{min}','(DJF mean removed)'},'fontsize',16);
xlim([-pi pi]); ylim([0 pi/2]);  % tightmap??

savefig(gcf,['quantile_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['quantile_',textTH,'_',text,'.pdf'])  %xtrmfreq
%}

%% time shape, xtrm_scatter_pchan
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];

load(['temp_',textTH,'_',text,'.mat'],'ver','caseid','text','textTH','thresh' ,'ds','latt42','lont42','yStart','yEnd','nyr','time' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend','lsm_jja' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend','lsm_djf')

fn_tend = ['../ncl/daily6h_',caseid,'.nc'];

T850  = ncread(fn_tend,'T850Daily');
W850  = ncread(fn_tend,'W850Daily');
T850z = ncread(fn_tend,'S850Daily');
T850AdvU = ncread(fn_tend,'T850AdvU');
T850AdvV = ncread(fn_tend,'T850AdvV');
T850AdvW = ncread(fn_tend,'T850AdvW');
T850err  = ncread(fn_tend,'T850err');

%% check lat lon, referencing file0 (z500)
% time
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
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
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
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  T850_djf(:,:,tpointer+(0:tend-tstart)) = T850(:,:,tstart:tend);
  W850_djf(:,:,tpointer+(0:tend-tstart)) = W850(:,:,tstart:tend);
  T850z_djf(:,:,tpointer+(0:tend-tstart)) = T850z(:,:,tstart:tend);
  T850AdvU_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvU(:,:,tstart:tend);
  T850AdvV_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvV(:,:,tstart:tend);
  T850AdvW_djf(:,:,tpointer+(0:tend-tstart)) = T850AdvW(:,:,tstart:tend);
  T850err_djf(:,:,tpointer+(0:tend-tstart)) = T850err(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
clear T850 W850 T850z T850AdvU T850AdvV T850AdvW T850err

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
Hot_cp(:,latt42(:)<=30,:,:) = false;
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
%plot( -6:6,HotAdvU_t*86400,'c-o', -6:6,HotAdvV_t*86400,'b-v', -6:6,HotAdvW_t*86400,'r-o', -6:6,(HotAdvU_t+HotAdvV_t+HotAdvW_t)*86400,'k-o', -6:6,(Hoterr_t)*86400,'g-o')
%legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal 3d Adv','anomal Error');
plot( -6:6,HotAdvU_t*86400,'c-o', -6:6,HotAdvV_t*86400,'b-v', -6:6,HotAdvW_t*86400,'r-o', -6:6,Hoterr_t*86400,'g-o', -6:6,(HotAdvU_t+HotAdvV_t+HotAdvW_t+Hoterr_t)*86400,'k-o')
legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal Residue','anomal total');
xlabel('time (day)');ylabel('T850 tendency (K/day)');grid on;

%savefig(gcf,['HotTimeEvolution_',textTH,'_',text,'.fig'])
%print(gcf,'-dpdf',['HotTimeEvolution_',textTH,'_',text,'.pdf'])
print(gcf,'-dpdf',['HotTimeEvolution_',ver,'.pdf'])


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
Cold_cp(:,latt42(:)<=30,:,:) = false;
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
%plot( -6:6,ColdAdvU_t*86400,'c-o', -6:6,ColdAdvV_t*86400,'b-v', -6:6,ColdAdvW_t*86400,'r-o', -6:6,(ColdAdvU_t+ColdAdvV_t+ColdAdvW_t)*86400,'k-o', -6:6,(Colderr_t)*86400,'g-o')
%legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal 3d Adv','anomal Error','location','southeast');
plot( -6:6,ColdAdvU_t*86400,'c-o', -6:6,ColdAdvV_t*86400,'b-v', -6:6,ColdAdvW_t*86400,'r-o', -6:6,Colderr_t*86400,'g-o', -6:6,(ColdAdvU_t+ColdAdvV_t+ColdAdvW_t+Colderr_t)*86400,'k-o')
legend('anomal U Adv','anomal V Adv','anomal W Adv','anomal Residue','anomal total','location','southeast');
xlabel('time (day)');ylabel('T850 tendency (K/day)');grid on;

%savefig(gcf,['ColdTimeEvolution_',textTH,'_',text,'.fig'])
%print(gcf,'-dpdf',['ColdTimeEvolution_',textTH,'_',text,'.pdf'])
print(gcf,'-dpdf',['ColdTimeEvolution_',ver,'.pdf'])
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
plot(latt42,krAdvU_y,'c-o',latt42,krAdvV_y,'b-v',latt42,krAdvW_y,'r-o',latt42,krErr_y,'g-o')
legend('U Adv','V Adv','W Adv','Error');
xlabel('Latitude (deg)'); ylabel('regress to T anomaly (day^{-1})'); grid on
title('JJA');

subplot(2,1,2)
plot(latt42,kcAdvU_y,'c-o',latt42,kcAdvV_y,'b-v',latt42,kcAdvW_y,'r-o',latt42,kcErr_y,'g-o')
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
plot(latt42,krAdvU_y,'c-o',latt42,krAdvV_y,'b-v',latt42,krAdvW_y,'r-o',latt42,krErr_y,'g-o')
legend('U Adv','V Adv','W Adv','Error');
xlabel('Latitude (deg)'); ylabel('regress to T anomaly (day^{-1})'); grid on
title('DJF');

subplot(2,1,2)
plot(latt42,kcAdvU_y,'c-o',latt42,kcAdvV_y,'b-v',latt42,kcAdvW_y,'r-o',latt42,kcErr_y,'g-o')
legend('U Adv','V Adv','W Adv','Error');
xlabel('Latitude (deg)'); ylabel('regress to T tendency (unitless)'); grid on

savefig(gcf,['TendencyRegressDJF_',textTH,'_',text,'.fig'])
print(gcf,'-dpdf',['TendencyRegressDJF_',textTH,'_',text,'.pdf'])
%}

%% polyfit / plot
%{
thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
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
  areaEarth(latt42(:)<=thresh{5})=0;

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

thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
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


% SeasonalCycle, Area
%{
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);

thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
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
%plot([1 nd_jja],HotQuantile(i,j)*[1 1],'k-');
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
%plot([1 nd_jja],ColdQuantile(i,j)*[1 1],'k-');
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

%% case study
%{
clear;
thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
ver=['x919_',caseid];  %
load(['temp_',ver,'.mat']);
disp('finish load'); toc

fn_z500a  = ['../index_wise/Z500_09xx_',caseid,'.mat'];
mat_z500a = matfile(fn_z500a);
%Z500a = mat_z500a.ZaDaily;  % already in latt42/lat0
Z500 = mat_z500a.Z500Daily;  %already in latt42/lat0
Z500a_jja = mat_z500a.Wgt_jja;
Z500a_djf = mat_z500a.Wgt_djf;

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
%

%% collect JJA
timeNan = [];

Z500_jja = zeros(ds_jja,'single');
Z850_jja = zeros(ds_jja,'single');
T850_jja = zeros(ds_jja,'single');
W850_jja = zeros(ds_jja,'single');
T850z_jja = zeros(ds_jja,'single');
T850AdvU_jja = zeros(ds_jja,'single');
T850AdvV_jja = zeros(ds_jja,'single');
T850AdvW_jja = zeros(ds_jja,'single');
T850err_jja = zeros(ds_jja,'single');
%PER7050_jja = false(ds_jja);
%Wgt7050_jja = zeros(ds_jja,'single');
%PESR2631_jja = zeros(ds_jja,'single');
%E2631_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  Z500_jja(:,:,tpointer+(0:tend-tstart)) = Z500(:,:,tstart:tend);
  Z850_jja(:,:,tpointer+(0:tend-tstart)) = Z850(:,:,tstart:tend);
  T850_jja(:,:,tpointer+(0:tend-tstart)) = T850(:,:,tstart:tend);
  W850_jja(:,:,tpointer+(0:tend-tstart)) = W850(:,:,tstart:tend);
  T850z_jja(:,:,tpointer+(0:tend-tstart)) = T850z(:,:,tstart:tend);
  T850AdvU_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvU(:,:,tstart:tend);
  T850AdvV_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvV(:,:,tstart:tend);
  T850AdvW_jja(:,:,tpointer+(0:tend-tstart)) = T850AdvW(:,:,tstart:tend);
  T850err_jja(:,:,tpointer+(0:tend-tstart)) = T850err(:,:,tstart:tend);
%  PER7050_jja(:,:,tpointer+(0:tend-tstart)) = PER7050(:,:,tstart:tend);
%  Wgt7050_jja(:,:,tpointer+(0:tend-tstart)) = Wgt7050(:,:,tstart:tend);
%  PESR2631_jja(:,:,tpointer+(0:tend-tstart)) = PESR2631(:,:,tstart:tend);
%  E2631_jja(:,:,tpointer+(0:tend-tstart)) = E2631(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
clear PER7050 Wgt7050 Z500 Z850 PESR2631 E2631  T850 W850 T850z T850AdvU T850AdvV T850AdvW T850err

%% remove trend
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
%

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

%Hotid_jja(Hotid_jja==0) = nan;
addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added
%rng(2017);

fn_figure = ['case10.ps'];
%fn_figure = ['case',datestr( f_h2d(mean(time_jja(ind))),'yyyymmdd'),'.ps'];
system(['rm ',fn_figure]);

% area time series
%{
figure('units','inches','position',[0 1 12 9], 'paperUnits','inches','papersize',[12 9],'paperposition',[0 0 12 9]);

%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%textTH = [textTH, '_org'];
%textTH = [textTH, '_wgt'];

%wrk = (hours(mean(time_jja(ind)))+datetime('1900-01-01 00:00:0.0','InputFormat','yyyy-MM-dd HH:mm:s.S'));
%yyyy = wrk.Year;
yyyy = ceil(mean(ind)/nd_jja);  % index starting 1
wrk = (yyyy-1)*nd_jja +(1:nd_jja);
%yyyy=2014;
%wrk=(yyyy-yStart)*nd_jja+(1:nd_jja);
%figure; plot(wrk,bjjaArea_t(wrk),'k-', wrk,hotArea_t(wrk),'r-');

%plot(mod(ind,nd_jja),bjjaArea_t(ind));

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

%for iii = ind(:)'  % row vector
%    [mh t] = ind2sub(size(bArea_ht), iii);
%    if (any(any(Hot_jja(:,:,iii) & repmat(areaEarth(:)'>0, [ds(1) 1]) )))
for m = 1:10
  for iii = HotAttr.tstart(csort(m)):HotAttr.tstart(csort(m))+nnz(HotAttr.areat{csort(m)})-1
%for iii = 1:nd_jja
%HotAttr.tmax(csort(m))
clf;

%pcolor
%{
%subplot(3,3,3); hold off;
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
%contourm(latt42,lonplot,double(Hotid_jja([1:end,1],:,iii))',[0.5 6.5]); axis equal tight;
pcolormPC(latt42,lont42,Hotid_jja(:,:,iii)');
hold on;
plotm(coastlat,coastlon,'color',[0.5 0.5 0.5])
colormap(gca,jet); caxis([-0.5 9.5]); colorbar; caxis auto;

%contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-');
%title({'Hotid (color)'});
title({['Hotid (color) ',datestr( f_h2d(time_jja(iii)) )]});
tightmap;
%}

subplot(3,3,1); hold off;
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
%contourm(latt42,lonplot,double(mx2t_jja([1:end,1],:,iii))',[15:5:30]); axis equal tight;
contourm(latt42,lonplot,double(mx2t_jja([1:end,1],:,iii))',[4:2:10],'linewidth',2); axis equal tight;
%colormap(gca,jet(4)); caxis([12.5 32.5]); %colorbar;
colormap(gca,jet); caxis([4 10]); %colorbar;
%title({'mx2t (color)',['iii=',num2str(iii)]});
title({['mx2t (color) ',datestr( f_h2d(time_jja(iii)) )]});
tightmap;

%template
%{
subplot(3,3,4); hold off;
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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

subplot(3,3,4); hold off;
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
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
axesm('MapProjection','pcarree','MapLatLimit',latt42(round(HotAttr.y(csort(m))))+[-25 25],'MapLonLimit',lont42(round(HotAttr.x(csort(m))))+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
plotm(coastlat,coastlon,'color',[0.7 0.7 0.7]);
hold on;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-','linewidth',2);
contourm(latt42,lonplot,double(86400*T850err_jja([1:end,1],:,iii))',[-6:2:-2,2:2:6],'linewidth',2); axis equal tight;
%contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',[-200:50:400],'showtext','on'); axis equal tight;
colormap(gca,jet); %caxis([4950 5850]); %colorbar;
title({'T850err (color)'});
tightmap;

%{
subplot(3,3,5); hold off;
axesm('MapProjection','pcarree','MapLatLimit',[30 80],'MapLonLimit',lont42(round(tmp(3)))+[-50 50]);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
contourm(latt42,lonplot,double(Z500a_jja([1:end,1],:,iii))',1.5*[96.7 123.7]); axis equal tight;
hold on;
plotm(coastlat,coastlon,'color',[0.5 0.5 0.5])
colormap(gca,jet(4)); caxis([150 180]); %colorbar;
contourm(latt42,lonplot,double(Hot_jja([1:end,1],:,iii))',[0.5,0.5],'k-');
title({'Z500a (color)'});
tightmap;

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
%
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

%
end  % loop csort
%    end
%end

system(['ps2pdf ',fn_figure]);

%}

%% functions
function rneg = rneg_scherrer(x)
%  addpath('../index_wise');
  verX='x919';
  caseid=['ERA-interim_19790101-20171231'];
  clear prm rrr
  rrr.r02=-99;
%  prm.D = min(max(round(x(1)-100),1),90);
  prm.D = 1+abs(round(x(1)-100));
  prm.GHGNT = x(2)-100;
  prm.GHGST = x(3)-100;
%  prm.dphi = min(max(round(x(4)-50),1),10);
  prm.dphi = 1+abs(-10+mod(10+round(x(4)-50),20));
  strTitle = sprintf('4a26: S06, 40N-, D=%i,GHGNT=%g,GHGST=%g,dphi=%i',prm.D,prm.GHGNT,prm.GHGST,prm.dphi);
  BlockStat_scherrer_pchan05;
  rneg = - corr(PERljja_n(:),Hot_n(:));
end

%:%s?colormap(\([^g]\)?colormap(gca,\1?gc
%:%s/num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E'/sprintf('%.0fN %.0fE',latt42(j),lont42(i))/gc
 %: %s?'\.\./matlab/scatter_?'../index_wise/scatter_?gce | %s?'BlockFreq_?'../index_wise/BlockFreq_?gce | %s?'Var_?'../index_wise/Var_?gce | %s?'all?'../index_wise/all?gce | %s?'tune_?'../index_wise/tune_?gce | %s?\.\./pedram?../index_wise?gce | %s?\.\./matlab/??gce | %s?\.\./ncl/??gce
 %: g/clearvars/s/Hot_n/& Hot_xyn/gc

 %:'<,'>s/^/  /gce | '<,'>s/emin\|ind\>\|\<m.\>/&(nr)/gce | '<,'>s/rrr.e01(/&nr,/gce | '<,'>s/(nr,:)/&'/gce | '<,'>s/jja:/& %s/gce | '<,'>s/',/&regArr{nr},/gce | '<,'>s/strTitle/titleArr{nr}/gce | noh
 %:'<,'>s/(ind\>/&(2)/gce | '<,'>s/rrr.e01(/&nr,/gce | '<,'>s/\<m.\>/&(nr)/gce | '<,'>s/,ind\>/&(nr)/gce | '<,'>s/e01arr(:,/rrr.&nr,/gce | '<,'>s/strTitle/titleArr{nr}/gce | noh

 %: 1742,$g/^pcolorPH/norm Ofor nr=1:size(mask_xyr,3)
 %: 1742,$g/^pcolorPH/norm Osubplot(2,2,nr);
 %: 1742,$g/^set(gca/norm oend % nr nRegion loop
 %: %s/,'rrr\([^']*\)'/&,'rbig\1'/gce
 %: g/rrr.PERjja_npq(:, rrr.np/norm O      rrr.PERjja_xynpq(:,:,:, rrr.np,rrr.nq,rrr.ns) = squeeze(mean( reshape(PER_jja, ds(1),ds(2),[],nyr ), 3));
 %blockstat: g/temp_/s/'ColdAttr'/& ,'Hot_xyn'/gc

%:'<,'>s/jja\C/djf/gce | '<,'>s/JJA\C/DJF/gce | '<,'>s/Hot\C/Cold/gce | '<,'>s/hot\C/cold/gce | '<,'>s/mx2t/mn2t/gce | '<,'>s/yStart:yEnd/yStart+1:yEnd/gce | '<,'>s/yyyy-yStart+1/yyyy-yStart/gce | '<,'>s?/nyr?/(nyr)?gce | '<,'>s/nyr/&-1/gce | noh

%:'<,'>s/nccreate(fn_savenc,'\([^']*\)',.*$/ncwrite(fn_savenc,'\1',\1)/ | noh

% : set fdm=expr foldexpr=getline(v\:lnum)=~'^%%.*$'?0\:1:
% vim: set fdm=marker foldmarker=%{,%}:

