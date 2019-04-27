%%Interpolates nc files output from ncrcat, output as T42 for e.bindex.exe
% previously known as bindex_pre_intp_pchan.m / bindex_read_pchan.m
%%Only for reanalysis, use bindex_fms_pchan.m for fms.
% TODO 06.m: label contiguous, 15/15day*11yr, normalize by sin lat
%%see matlab_submit_block.ctl on how to run the script
% 43s for holyjacob06 to get rrr

% To load this blocking index:
%{
ver=['0a13_',caseid];  %NewReversal, bugfix+2
fn_load2 = ['../index_wise/BlockFreq_',ver,'.mat'];
mat_load2 = matfile(fn_load2);
PER0a13 = (mat_load2.PER);
%PER0a13 = (mat_load2.label>0);
%}
 %load(fn_load2,'timeNan','PER')
% PER = single(PER>=5);  PER(:,:,timeNan)=nan;
% 1=blocking, 0=no-blocking, nan=undefined

%% load and save data
tic;
oldpwd=pwd;

%
season=true;
%caseid='historical_MIROC-ESM-CHEM_19660101-20051231';
%caseid='NCEP-R2_19790101-20141231';
%caseid='ERA-40_19580101-20011231';
% caseid='ERA-interim_19790101-20171231'; season=true;
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
%load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','ds','Cold_yxht','Hot_yxht','T850f_yxht','areaEarth')
%load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh' ,'ds','yStart','yEnd','nyr' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend','lsm_jja' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend','lsm_djf')
%load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh' ,'ds','yStart','yEnd','nyr','f_h2d' ,'ds_jja','hJJAstart','hJJAend','lsm_jja' ,'ds_djf','hDJFstart','hDJFend','lsm_djf')
load(['temp_',verX,'_',caseid,'.mat'],'caseid','thresh' ,'ds','yStart','yEnd','nyr','f_h2d','latt42','areaEarth' ,'ds_jja','nd_jja','hJJAstart','hJJAend','lsm_jja','area_jja' ,'ds_djf','nd_djf','hDJFstart','hDJFend','lsm_djf')

%prm.dphi = 5;  %dphi ~= 14deg (15)
prm.yS1  = 1;  % 87.86deg
prm.yS2  = find(latt42<-thresh{5},1,'last');  %18; %40.46deg  %21; % 32.09deg  %
prm.yN1  = 64+1-prm.yS2;
prm.yN2  = 64+1-prm.yS1;
%prm.latyday = 5;  % 14deg (13.5)
%prm.lonyday = 6;  % 16.875deg (18)
%prm.latonset = 7;  % 19.53deg (20.25)
%prm.lononset = 10;  % 28.125deg (27)

%prm.GHGST = 0;
%prm.GHGNT = -10.0; %Inf;  % ( m / deg lat )
%prm.A = 1.5;
prm.O = 0.5;
%prm.S = 2.5e6;
%prm.D = 5;
%prm.R = 1;

yS1=prm.yS1; yS2=prm.yS2; yN1=prm.yN1; yN2=prm.yN2;

%ver=['0a13_',caseid];  %
ver=[strTitle(1:4),'_',caseid];  %
 % ~/script/blocking/verlist.txt
 % 0000 z/sin(lat), D=5,A=1.5
 % b01 D=7,A=1.5
 % b02 D=5,A=1.5, bugfix on lat
 % b03 D=7,A=1.5, bugfix on lat
 % b05 D=7,A=2,
 % b06 D=14,A=1.5,
 % 03xx z,
 % 05xx fms, no-season
 % 06xx JJA mean
 % 07xx DJF mean
%strTitle = '0a02: D13, z/sin(lat), 15/15day*11yr';  %'Dunn-Sigouin and Son 2013 index';
%strTitle = '0a10: D13, psi500, D=5,32-90N,S=2.5e6,A=1.5 15/15day*11yr';  %'Dunn-Sigouin and Son 2013 index';
%strTitle = '0a12: D13, psi500, D=5,40-90N,S=2.5e6,A=1.5 15/15day*11yr';  %'Dunn-Sigouin and Son 2013 index';
%strTitle = '0a13: D13, psi500, D=5,astd40-90N,S=2.5e6,A=1.5 15/15day*11yr';  %'Dunn-Sigouin and Son 2013 index';
%strTitle = '0b02: D13, z250/sin(lat), 15/15day*11yr';  %'Dunn-Sigouin and Son 2013 index';

%
 fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
latt42  = ncread(fn_t42,'lat');
lont42  = ncread(fn_t42,'lon');

%% load nco*.nc
ver0=ver; ver0(1)='0'; ver0(3:4)='xx';
if (exist(['../sks/int_z500_zg_day_',caseid,'N.nc'],'file')==2 && exist(['../index_wise/Var_',verX,ver0,'.mat'],'file')==2)
load(['../index_wise/Var_',verX,ver0,'.mat'],'Wgt','Wgt_jja','Wgt_djf','VarDaily','Wgtjjasd','Wgtdjfsd','time');  % TODO 29xx given in BlockStat_d2d_pchan02.m

%
else
 % ncrcat -d plev,50000.0 zg_day_MRI-CGCM3_historical_r1i1p1_*nc int_z500_zg_day_MRI-CGCM3_historical_r1i1p1_19660101-20051231.nc
 % ncrcat -d level,500.0 ../NCEP-R2/hgt.*.nc ../sks/nco_z500_zg_day_NCEP-R2_19790101-20141231.nc
 % ncap2 -v -s "zg=float(hgt)" ../sks/nco_z500_zg_day_NCEP-R2_19790101-20141231.nc ../sks/nco2_z500_zg_day_NCEP-R2_19790101-20141231.nc

 % ~/sw/ecmwfapi/wget.ecmwf.txt: ncap2 -s "z@scale_factor=float(z@scale_factor)/9.81f; z@add_offset=float(z@add_offset)/9.81f;" z250_00Z_ERA-interim_19790101-20151231.nc nco_z250_00Z_ERA-interim_19790101-20151231.nc
%fn_nco   = ['../ERA-interim/nco_z500_zg_day_',caseid,'.nc'];  % 2a
%VarDaily = single(ncread(fn_nco,'z'));
fn_nco   = ['../ERA-interim/zg_day_',caseid,'.nc'];  % 2a
VarDaily = single(ncread(fn_nco,'z'))/9.81;
VarDaily = squeeze(VarDaily);
disp('finish load'); toc
%  ds = size(VarDaily);  % load from temp.mat
%  nyr=round(ds(3)/365.25);

%fn_load0 = ['../sks/SKS_int_z500_',caseid,'.dat'];  % SKSanomaly@main
%fn_load1 = ['../sks/thresholds_SKS_int_z500_',caseid,'.dat'];  % SKSanomaly@threshold
%fn_load3 = ['../sks/label_int_z500_',caseid,'.dat'];  % durthresh
%fn_load2 = ['../sks/dfreq_int_z500_',caseid,'.dat'];  % fdist
%fn_load4 = ['../sks/annualcycle_int_z500_',caseid,'.dat'];  % annualcycle
%fn_load5 = ['../sks/ddur_int_z500_',caseid,'.dat'];   % ddist
% fn_save  = ['../index_wise/Z500_',caseid,'.mat'];
%fn_savenc = ['../sks/int_z500_zg_day_',caseid,'.nc'];  % NH, SH later

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

%  for t = 1:ds(3)
%    system(['echo -ne "\r',num2str(t),'"']);
%    VarDaily(:,:,t) = interp2(lat(end:-1:1),lon,hgtnco(:, end:-1:1, t),latt42',lont42,'linear');
%    % make it increasing?
%  end
% whos VarDaily
% size(VarDaily)

%  Z500sd=std(Z500Daily,[],3);  % do after removing seasonal cycle
  x = lont42;
  y = latt42;

%% output ../sks/int_z500_zg*nc, aka. data11.nc  raw Z500 for reversal %TODO
 % increasing lat, rename z, z/9.8
%x{
%save(fn_save,'Z500Daily','x','y','Z500sd');  % suppress output before removing seasonal cycle, done in bindex_read_pchan.m

fn_savenc = ['../sks/int_z500_zg_day_',caseid,'N.nc'];
% system(['rm ',fn_savenc]);
system(['ncks -Ov lat,lat_bnds,lon,lon_bnds ',fn_t42,' ',fn_savenc]);
system(['ncks -A -v time ',fn_nco,' ',fn_savenc]);
%system(['ncks -A -v time,time_bnds ',fn_nco,' ',fn_savenc]);
nccreate(fn_savenc,'zg','Dimensions',{'lon',128,'lat',64,'time',ds(3)},'DataType','single','Format','classic')
%ccreate(fn_savenc,'lat','Dimensions',{'lat',64},'DataType','double','Format','classic')
%ccreate(fn_savenc,'lon','Dimensions',{'lon',128},'DataType','double','Format','classic')
%ccreate(fn_savenc,'time','Dimensions',{'time',ds(3)},'DataType','double','Format','classic')
ncwrite(fn_savenc,'zg',VarDaily)
%cwrite(fn_savenc,'lat',latt42)
%cwrite(fn_savenc,'lon',lont42)
%cwrite(fn_savenc,'time',time)

fn_savenc = ['../sks/int_z500_zg_day_',caseid,'S.nc'];
 system(['cp -f ../sks/int_z500_zg_day_',caseid,'N.nc ',fn_savenc]);
ncwrite(fn_savenc,'zg',VarDaily(:,end:-1:1,:))
disp('finish int*nc'); toc
%x}

%% nan, sign, angle, demean, movmean
%timeNan = ds(3);  % 700?, 702?, 703?
%VarDaily(:,:,timeNan) = nan;
  Wgt = VarDaily;
%  clear VarDaily

%  Wgt = Wgt .* repmat(reshape(sign(latt42), [1 ds(2) 1]), [ds(1) 1 ds(3)*4]);  %705?

%  Wgt = -Wgt;  % 7060
%  Wgt = Wgt*cosd(prm_angle) + U850Daily*sind(prm_angle);  % 7060

%  wrk  = nanmean( reshape(permute(double(Wgt),[2 1 3]), ds(2),[]) ,2); %xyt->yxt, double precision needed
%  Wgt = Wgt - repmat( reshape(wrk, [1 ds(2) 1]), [ds(1) 1 ds(3)]);

% no time movmean

%% x8xx: mimic SKSanomaly.f90
%{
Wgt_bar = movmean(Wgt,365,3,'Endpoints','fill');
Wgt_bar(:,:,1:182) = repmat( Wgt_bar(:,:,183), [1 1 182]);
Wgt_bar(:,:,end-181:end) = repmat( Wgt_bar(:,:,end-182), [1 1 182]);
%Wgt = Wgt - Wgt_bar;
Wgt_star = movmean(Wgt - Wgt_bar, 31,3);  % no Endpoints treatment for star..
Wgt_hat = nan([ds(1:2) 366],'single');

for t = 1:366
  tArr = days( datetime('0000-01-01')+caldays(t-1)+calyears(yStart:yEnd) - f_h2d(time(1)) )+1; % order of addition matters
  Wgt_hat(:,:,t) = mean(Wgt_star(:,:,tArr),3);
end
dtArr = f_h2d(time); %DArr = days(dtArr - dateshift(dtArr,'start','year') )+1;
dtArr.Year=0; DArr = days(dtArr - datetime('0000-01-01') )+1;
Wgt = Wgt - Wgt_bar - Wgt_hat(:,:,DArr);  % prime
WgtCli = Wgt_bar + Wgt_hat(:,:,DArr);
clear Wgt_bar Wgt_star Wgt_hat tArr dtArr DArr
%}

%% x9xx: mimic SKSanomaly.f90
%
Wgt_star = movmean(Wgt, 15,3);  % no Endpoints treatment for star..
Wgt_star = movmean(Wgt_star, 15,3);  % no Endpoints treatment for star..
Wgt_hat = nan([ds(1:2) 366 nyr],'single');

for t = 1:366
  tArr = days( datetime('0000-01-01')+caldays(t-1)+calyears(yStart:yEnd) - f_h2d(time(1)) )+1; % order of addition matters
  Wgt_hat(:,:,t,:) = Wgt_star(:,:,tArr);
end

Wgt_hat(:,:,[1:104,end-75:end]) = nan;  % jump elsewhere not Jan 1..
Wgt_hat = movmean(Wgt_hat,11,4,'omitnan');

dtArr = f_h2d(time); DArr = 366*(dtArr.Year-yStart);
dtArr.Year=0; DArr = DArr + days(dtArr - datetime('0000-01-01') )+1;
Wgt = Wgt - Wgt_hat(:,:,DArr);  % prime
Wgt(isnan(Wgt)) = 0;
WgtCli = Wgt_hat(:,:,DArr);
clear Wgt_bar Wgt_hat tArr dtArr DArr  %Wgt_star
%

%% collect JJA
Wgt_jja = zeros(ds_jja,'single');
WgtCli_jja = zeros(ds_jja,'single');
WgtStar_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  Wgt_jja(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);
  WgtCli_jja(:,:,tpointer+(0:tend-tstart)) = WgtCli(:,:,tstart:tend);
  WgtStar_jja(:,:,tpointer+(0:tend-tstart)) = Wgt_star(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
Wgt_djf = zeros(ds_djf,'single');
WgtCli_djf = zeros(ds_djf,'single');
WgtStar_djf = zeros(ds_djf,'single');
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  Wgt_djf(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);
  WgtCli_djf(:,:,tpointer+(0:tend-tstart)) = WgtCli(:,:,tstart:tend);
  WgtStar_djf(:,:,tpointer+(0:tend-tstart)) = Wgt_star(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
clear WgtCli Wgt_star

Wgtjja_xyn = squeeze(mean( reshape(WgtCli_jja, ds(1),ds(2),[],nyr ), 3));  % x,y,yr

Wgtdjf_xyn = squeeze(mean( reshape(WgtCli_djf, ds(1),ds(2),[],nyr-1 ), 3));  % x,y,yr

%% x6xx: remove JJA mean
%{
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

%Wgt = nan(ds,'single');
Wgt = zeros(ds,'single');
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
%  zprime = Z500Daily - repmat( mean(Wgt_jja,3), [1 1 ds(3)]);
%  Wgt_jja = Wgt_jja - repmat( mean(Wgt_jja,3), [1 1 ds_jja(3)]);
%}

disp('start saving'); toc
% calculate sd
  Wgtjjasd=std(Wgt_jja,[],3);  % never 'normalize by f'
  Wgtdjfsd=std(Wgt_djf,[],3);  % never 'normalize by f'

%  if ( strcmp(sksid(1:2),'00') || strcmp(sksid(1:2),'01') || strcmp(sksid(1:2),'04') )
  if (any( strcmp(ver(1:2),{'00','01','04','0a','2a','2b'}) ))  %TODO 0a
    for j=1:ds(2)
        Wgt(:,j,:) = Wgt(:,j,:)./abs(sin(latt42(j)*pi/180.)) *sin(pi/4);
        Wgt_jja(:,j,:) = Wgt_jja(:,j,:)./abs(sin(latt42(j)*pi/180.)) *sin(pi/4);
        Wgt_djf(:,j,:) = Wgt_djf(:,j,:)./abs(sin(latt42(j)*pi/180.)) *sin(pi/4);
    end
  end

%% output for Hassanzadeh et al. 2014 %TODO
%
if (season)
 ver0=ver; ver0(1)='0'; ver0(3:4)='xx';
% ver0=['0axx_',caseid];
% fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
 fn_save  = ['../index_wise/Var_',verX,ver0,'.mat'];

%ZaDaily = Wgt;
%Z500sd=std(ZaDaily,[],3);  % use JJA instead for BlockStat_Hybrid_pchan05
%  if ( strcmp(ver0(1:2),'0a') )
%    Z500sd = Wgtjjasd;
%  elseif ( strcmp(ver0(1:2),'07') )
%    Z500sd = Wgtdjfsd;
%  end

%Z500Daily = ncread(fn_int,'zg');
%y        = ncread(fn_t42,'lat');
%x        = ncread(fn_t42,'lon');

% save(fn_save,'ZaDaily','Z500Daily','x','y','Z500sd','-v7.3');
 save(fn_save,'Wgt','Wgt_jja','Wgt_djf','VarDaily','latt42','lont42','Wgtjjasd','Wgtdjfsd','time','-v7.3');
% disp('../index_wise/Var_*mat saved, return'); return;

% ver=[ver];
% caseid=ver; caseid(1:5)=[];
% clearvars -except   caseid ver season nyr ds yStart yEnd Wgtjjasd Wgtdjfsd   ZaDaily VarDaily x y Z500sd

% cd ../index_wise
% BlockStat_Hybrid_pchan04
% BlockStat_Hybrid_pchan05
% BlockStat_Hybrid_pchan06

else
 warning('unexpected season'); return;
end
% end Hassanzadeh
%
end  % file exist

disp('finish preproc'); toc

%% tracking algorithm see BlockStat_m2d_pchan.m
% label contiguous, see blocklabel, m2d

%% create nc for program  % see sch*.m
 % D2D and Hassanzadeh are the only two that use 1.5 sigma
 % using sigma will then have all problem of JJA/DJF, NH/SH (2x2=4)
 % VarDaily in int*nc is good for all
 % Wgt(zprime) is good for all
 % sdJJA_NH is only good for JJA+NH
 % Wgtjjasd is only good for JJA

% JJA
% ver=['0axx_',caseid];
% caseid=ver; caseid(1:5)=[];
 sksid=ver;  sksid(3:4)=[];

% NH
%  std_sks = std(reshape( double(Wgt_jja(:,[44:64],:)) ,[],1));  % llim=44,ulim=64
%  std_sks = std(reshape( double(Wgt_jja(:,[yN1:yN2],:)) ,[],1));  % llim=44,ulim=64
%  sdJJA_NH = std_sks;

% astd: 0x13+, 2x68+, 2x71+
Wgtjjasd=std(double(Wgt_jja),[],3);
%Wgtdjfsd=std(double(Wgt_djf),[],3);
%  mean(mean(Wgtjjasd(:,[yN1:yN2]))), sqrt(mean(mean(Wgtjjasd(:,[yN1:yN2]).^2)))
sdJJA_NH = sqrt(areaEarth(yN1:yN2)*mean(Wgtjjasd(:,[yN1:yN2]).^2)'/sum(areaEarth(yN1:yN2)));
%sdJJA_SH = sqrt(areaEarth(yS1:yS2)*mean(Wgtjjasd(:,[yS1:yS2]).^2)'/sum(areaEarth(yS1:yS2)));sdJJA_SH=sdJJA_NH;
%sdDJF_NH = sqrt(areaEarth(yN1:yN2)*mean(Wgtdjfsd(:,[yN1:yN2]).^2)'/sum(areaEarth(yN1:yN2)));
%sdDJF_SH = sqrt(areaEarth(yS1:yS2)*mean(Wgtdjfsd(:,[yS1:yS2]).^2)'/sum(areaEarth(yS1:yS2)));sdDJF_SH=sdDJF_NH;
std_sks = sdJJA_NH;
sdlJJA_NH = sqrt( sum(area_jja(:).*Wgtjjasd(:).^2)./sum(area_jja(:)) );
%sdlDJF_NH = sqrt( sum(area_jja(:).*Wgtdjfsd(:).^2)./sum(area_jja(:)) );

%
 fn_save0 = ['../sks/SKS_int_z500_',sksid,'N.dat'];  % SKSanomaly@main
 fn_save1 = ['../sks/thresholds_SKS_int_z500_',sksid,'N.dat'];  % SKSanomaly@threshold

fid0 = fopen(fn_save0,'w');
fwrite(fid0, Wgt, 'real*4');
fclose(fid0);

fid1 = fopen(fn_save1,'w');
fprintf(fid1, '%10.5f\n', std_sks*ones(12,1));
fclose(fid1);

%{
% SH
  std_sks = std(reshape( double(Wgt_jja(:,[1:21],:)) ,[],1));  % llim=44,ulim=64
  sdJJA_SH = std_sks;

 fn_save0 = ['../sks/SKS_int_z500_',sksid,'S.dat'];  % SKSanomaly@main
 fn_save1 = ['../sks/thresholds_SKS_int_z500_',sksid,'S.dat'];  % SKSanomaly@threshold

fid0 = fopen(fn_save0,'w');
fwrite(fid0, Wgt(:,end:-1:1,:), 'real*4');
fclose(fid0);

fid1 = fopen(fn_save1,'w');
fprintf(fid1, '%10.5f\n', std_sks*ones(12,1));
fclose(fid1);

% DJF
 ver=['07xx_',caseid];
% caseid=ver; caseid(1:5)=[];
 sksid=ver;  sksid(3:4)=[];

% NH
  std_sks = std(reshape( double(Wgt_djf(:,[44:64],:)) ,[],1));  % llim=44,ulim=64
  sdDJF_NH = std_sks;

 fn_save0 = ['../sks/SKS_int_z500_',sksid,'N.dat'];  % SKSanomaly@main
 fn_save1 = ['../sks/thresholds_SKS_int_z500_',sksid,'N.dat'];  % SKSanomaly@threshold

fid0 = fopen(fn_save0,'w');
fwrite(fid0, Wgt, 'real*4');  % same as jja
fclose(fid0);

fid1 = fopen(fn_save1,'w');
fprintf(fid1, '%10.5f\n', std_sks*ones(12,1));
fclose(fid1);

% SH
  std_sks = std(reshape( double(Wgt_djf(:,[1:21],:)) ,[],1));  % llim=44,ulim=64
  sdDJF_SH = std_sks;

 fn_save0 = ['../sks/SKS_int_z500_',sksid,'S.dat'];  % SKSanomaly@main
 fn_save1 = ['../sks/thresholds_SKS_int_z500_',sksid,'S.dat'];  % SKSanomaly@threshold

fid0 = fopen(fn_save0,'w');
fwrite(fid0, Wgt(:,end:-1:1,:), 'real*4');
fclose(fid0);

fid1 = fopen(fn_save1,'w');
fprintf(fid1, '%10.5f\n', std_sks*ones(12,1));
fclose(fid1);
%}

%disp(sprintf('jjaNH:%.2f jjaSH:%.2f djfNH:%.2f djfSH:%.2f',sdJJA_NH,sdJJA_SH,sdDJF_NH,sdDJF_SH));
%disp(sprintf('jjaNH:%.2f',sdJJA_NH));
disp(sprintf('jjaNH:%.2f jjaLand:%.2f ratio:%f',sdJJA_NH,sdlJJA_NH,sdJJA_NH/sdlJJA_NH));
%clear Wgt VarDaily
%disp('finish dat'); toc
%

clear Wgt VarDaily Wgtjjasd Wgtdjfsd
clear WgtStar_jja WgtStar_djf WgtCli_jja WgtCli_djf  % QC
%whos
%disp('return');return
%ver=['0a13_',caseid];
%TODO save(['../index_wise/wrk_',ver,'.mat'],'-v7.3');

%toc
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

% extra: (jja mean - annual mean) trend
mat_0602 = matfile(['../index_wise/wrk_0602_',caseid,'.mat']);
Wgt2jja_xyn = mat_0602.Wgtjja_xyn - Wgtjja_xyn;
Wgt2djf_xyn = mat_0602.Wgtdjf_xyn - Wgtdjf_xyn;

Weight_t = 1:nyr;
Weight_t = Weight_t - mean(Weight_t);
Weight_t = Weight_t / sumsqr(Weight_t);
Wgt2jja_trend = sum(Wgt2jja_xyn.*repmat(reshape(Weight_t,[1 1 nyr]),[ds(1:2) 1]),3);

Weight_t = 1:nyr-1;
Weight_t = Weight_t - mean(Weight_t);
Weight_t = Weight_t / sumsqr(Weight_t);
Wgt2djf_trend = sum(Wgt2djf_xyn.*repmat(reshape(Weight_t,[1 1 nyr-1]),[ds(1:2) 1]),3);

figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
subplot(1,2,1);
%axesm('MapProjection','vperspec','origin',[90 0],'MapParallels',6000,'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','ortho','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','eqdazim','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','breusing','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
contourfm(latt42,lonplot,double(Wgt2jja_trend([1:end,1],:)'),[-4:4]*0.5);  % TODO
%pcolormPC(latt42,lont42, Wgt2jja_trend'); shading flat;
%contourm(latt42,lonplot,double(Wgt2jja_trend([1:end,1],:)'),[-3:3]*0.5); %axis equal tight; %,'showtext','on'
%contourm(latt42,lonplot,double(Wgt2jja_trend([1:end,1],:)'),'showtext','on'); %axis equal tight; %,'showtext','on'
%colormap(jet(12)); caxis([0 12]); colorbar;
colormap(b2rPC(-5*0.5, 5*0.5, 10)); colorbar;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitle,'JJA trend - annual trend (unit/yr)'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
contourfm(latt42,lonplot,double(Wgt2djf_trend([1:end,1],:)'),[-4:4]*0.5);
%pcolormPC(latt42,lont42, Wgt2djf_trend'); shading flat;
%contourm(latt42,lonplot,double(Wgt2djf_trend([1:end,1],:)'),[-0.06:0.02:0.06]); %axis equal tight;
%colormap(jet(12)); caxis([0 12]); colorbar;
colormap(b2rPC(-5*0.5, 5*0.5, 10)); colorbar;
plotm(coastlat,coastlon,'k')
title({'DJF trend - annual trend (unit/yr)'},'fontsize',16);
tightmap;

print(gcf, '-dpsc2','-append',fn_figure);

% xyn: mean JJA/DJF time series for ramdom points: check movmean 5
%{
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
%}

%% x9xx: SeasonalCycle, Wgt, xtrm_reanalysis
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
    plot(reshape( squeeze(WgtStar_jja(i,j,:)),[],nyr ));
    plot(mean(reshape( squeeze(WgtStar_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%    plot([1 nd_jja],prm.A*[1 1],'k-');  % TODO
%    plot([1 nd_jja],jjaQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E']);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
%  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
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
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr-1);
    plot(reshape( squeeze(WgtStar_djf(i,j,:)),[],nyr-1 ));
    plot(mean(reshape( squeeze(WgtStar_djf(i,j,:)),[],nyr-1 ),2),'k','linewidth',1);
%    plot([1 90],prm.A*[1 1],'k-');  % TODO
%    plot([1 90],djfQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E']);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
%  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
%print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])

%% x9xx: SeasonalCycle, Wgt, xtrm_reanalysis
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
    plot(reshape( squeeze(WgtCli_jja(i,j,:)),[],nyr ));
    plot(mean(reshape( squeeze(WgtCli_jja(i,j,:)),[],nyr ),2),'k','linewidth',1);
%    plot([1 nd_jja],prm.A*[1 1],'k-');  % TODO
%    plot([1 nd_jja],jjaQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_jja(wrk))),hotArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E']);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
%  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
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
% xlabel
    ax=gca; ax.ColorOrder = jet(nyr-1);
    plot(reshape( squeeze(WgtCli_djf(i,j,:)),[],nyr-1 ));
    plot(mean(reshape( squeeze(WgtCli_djf(i,j,:)),[],nyr-1 ),2),'k','linewidth',1);
%    plot([1 90],prm.A*[1 1],'k-');  % TODO
%    plot([1 90],djfQuantile(i,j)*[1 1],'k-');
%plot(datenum(f_h2d(time_djf(wrk))),coldArea_t(wrk)); datetick('x','mm/dd'); grid on;
    title([num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E']);
    axis tight;
%    print(gcf, '-dpsc2','-append',fn_figure);
%  end  % TODO land or not
end
print(gcf, '-dpsc2','-append',fn_figure);  % TODO
%print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])

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
    plot([1 nd_jja],prm.A*sdJJA_NH*[1 1],'k-');  % TODO
%    plot([1 nd_jja],jjaQuantile(i,j)*[1 1],'k-');
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
    plot([1 90],prm.A*sdDJF_NH*[1 1],'k-');
%    plot([1 90],djfQuantile(i,j)*[1 1],'k-');
    title([num2str(int16(latt42(j))),'N ',num2str(int16(lont42(i))),'E']);
    axis tight;
%  end
end
print(gcf, '-dpsc2','-append',fn_figure);
%print(gcf,'-dpdf','-r600',['SeasonalCycleDJF_',textTH,'_',text,'.pdf'])

%system(['ps2pdf ',fn_figure]);
toc
%}

%
%clear VarDaily x y zmean Wgt Wgt_jja Wgt_djf %Z500sd
%clearvars -except   prm caseid ver season nyr ds yStart yEnd Wgtjjasd Wgtdjfsd

%% call the SKS program %TODO
%
tmpdir=['/scratch/',getenv('USER'),'/',getenv('SLURM_JOBID')];
mkdir(tmpdir);
if (isempty(ls(tmpdir)))
  system(['cp -ai ~/blocking/s02_SKS_HISTORICAL ~/blocking/b02_BLOCK_HISTORICAL ~/blocking/mlist99.txt ',tmpdir]);
end
%system(['rm -r ',tmpdir]);

% for FMS, assert maxday<45000
%{
cd ~/blocking/s02_SKS_HISTORICAL/
system(['sed -i ''s/\(nyr=\)[0-9]*,/\1',num2str(nyr),',/'' ~/blocking/s02_SKS_HISTORICAL/qq.fms.02.h']);
system(['sed -i ''s/\(maxday1[12]=\)[0-9]*,/\1',num2str(ds(3)),',/'' ~/blocking/s02_SKS_HISTORICAL/qq.fms.02.h']);
system(['ln -sf qq.fms.02.h qq.h']);
system(['sed -i ''s/\(^2[12] \).*\([NS]\)$/\1',caseid,'\2/'' ~/blocking/mlist99.txt']);

system(['./SKS.historical.pchan02.csh']);
cd ../b02_BLOCK_HISTORICAL/
system(['cp -a ../s02_SKS_HISTORICAL/qq.fms.02.h ./']);
system(['ln -sf qq.fms.02.h qq.h']);
system(['./e.bindex.historical.pchan02.csh']);
%}

cd([tmpdir,'/s02_SKS_HISTORICAL']);
 % for reanalysis,   %nyear, start..
system(['rm qq.h; cp -a qq.fms.02.h qq.h']);
system(['sed -i ''s/\(nyr=\)[0-9]*,/\1',num2str(nyr),',/'' qq.h']);
system(['sed -i ''s/\(maxday1[12]=\)[0-9]*\(,yrtype1[12]=\)./\1',num2str(ds(3)),'\21/'' qq.h']);
system(['sed -i ''s/\(^2[12] \).*\([NS]\)$/\1',caseid,'\2/'' ../mlist99.txt']);

 % When Matlab is executing, it changes LD_LIBRARY_PATH, causes linking errores. Instead of enumerate all -lnetcdf, etc., we change LD_LIBRARY_PATH back.
%setenv('LD_LIBRARY_PATH','/n/sw/fasrcsw/apps/Core/nco/4.5.3-fasrc01/lib64:/n/sw/fasrcsw/apps/Core/antlr/2.7.7-fasrc01/lib64:/n/sw/fasrcsw/apps/Core/udunits/2.2.18-fasrc01/lib64:/n/sw/fasrcsw/apps/Core/gsl/1.16-fasrc02/lib64:/n/sw/fasrcsw/apps/MPI/intel/15.0.0-fasrc01/impi/5.1.2.150-fasrc01/netcdf/4.1.3-fasrc09/lib:/n/sw/fasrcsw/apps/MPI/intel/15.0.0-fasrc01/impi/5.1.2.150-fasrc01/hdf5/1.8.12-fasrc12/lib64:/n/sw/fasrcsw/apps/Comp/intel/15.0.0-fasrc01/zlib/1.2.8-fasrc07/lib:/n/sw/intel-mpi-5.1.2.150/compilers_and_libraries_2016.1.150/linux/mpi/intel64/lib:/n/sw/intel-cluster-studio-2015/tbb/lib/intel64:/n/sw/intel-cluster-studio-2015/mkl/lib/intel64:/n/sw/intel-cluster-studio-2015/lib/intel64:/n/sw/matlab-R2017a/KNITRO:/n/sw/matlab-R2017a/sys/os/glnxa64:/n/sw/matlab-R2017a/extern/lib/glnxa64');
setenv('LD_LIBRARY_PATH',getenv('LD_LIBRARY_save'));

for m = 1:1
%sksVerArray = {'0a','07'};
sksVerArray = {ver(1:2)};

cd([tmpdir,'/s02_SKS_HISTORICAL']);
%ver(1:2) = sksVerArray{m};
% ver=['0202_',caseid];
setenv('sksVer',ver(1:2));
%system(['./SKS.historical.pchan02.csh']);

% ver(3:4) = '02';
cd ../b02_BLOCK_HISTORICAL/
setenv('bVer',ver(3:4));
setenv('DatDir',[oldpwd,'/../sks']);
system(['rm qq.h; cp -a ../s02_SKS_HISTORICAL/qq.h ./']);
%system(['ln -sf qq.fms.02.h qq.h']);
system(['sed -i "s/\(real::\s*sigma=\)[0-9*.-]*$/\1',num2str(prm.A),'/" e.bindex.F']);
system(['sed -i "s/\(real::\s*O=\)[0-9*.]*$/\1',num2str(prm.O),'/" e.bindex.F']);
system(['sed -i "s/\(integer::\s*S=\)[0-9*.]*$/\1',num2str(prm.S),'/" e.bindex.F']);
system(['sed -i "s/\(integer::\s*D=\)[0-9*.]*$/\1',num2str(prm.D),'/" e.bindex.F']);
%system(['sed -i "s/\(real::\s*sgrad=\)[0-9*.]*$/\1',num2str(0),'/" e.bindex.F']);
%system(['sed -i "s/\(real::\s*sgrad=\)[0-9*.\-]*$/\1',num2str(-99999),'/" e.bindex.F']);
if (prm.R==1)
system(['sed -i "/call GHG/,+1s/^./ /" e.bindex.F']);
else
system(['sed -i "/call GHG/,+1s/^./c/" e.bindex.F']);  %TODO
end
system(['./e.bindex.historical.pchan02.csh']);
if (ans~=0) cd(oldpwd); return; end

%% bindex_read
% clearvars -except   prm caseid ver season nyr ds yStart yEnd Wgtjjasd Wgtdjfsd
%cd(oldpwd)
%disp('b4 read'); toc
%system(['rsync -avu ~/script/blocking/bindex_read_pchan.m ./']);
% bindex_read_pchan;  % fread below!!
%disp('after bindex read'); toc

%% run all other index
%{
 clearvars -except   caseid ver season nyr ds yStart yEnd Wgtjjasd Wgtdjfsd   fn_int zprime
% caseid=ver; caseid(1:5)=[];

if (season)
 ver(3:4) = 'xx';
 fn_t42   = ['../sks/int_z500_zg_day_MIROC-ESM-CHEM_historical_r1i1p1_19660101-20051231.nc'];
 fn_save  = ['../index_wise/Z500_',ver,'.mat'];

ZaDaily = zprime;  clear zprime;  % given in bindex_read_pchan
%Z500sd=std(ZaDaily,[],3);  % use JJA instead for BlockStat_Hybrid_pchan05
  if ( strcmp(ver(1:2),'0a') )
    Z500sd = Wgtjjasd;
  elseif ( strcmp(ver(1:2),'07') )
    Z500sd = Wgtdjfsd;
  end

Z500Daily = ncread(fn_int,'zg');
y        = ncread(fn_t42,'lat');
x        = ncread(fn_t42,'lon');

 save(fn_save,'ZaDaily','Z500Daily','x','y','Z500sd','-v7.3');

% ver=[ver];
% caseid=ver; caseid(1:5)=[];
 clearvars -except   caseid ver season nyr ds yStart yEnd Wgtjjasd Wgtdjfsd   ZaDaily VarDaily x y Z500sd

% cd ../index_wise
% BlockStat_Hybrid_pchan04
 BlockStat_Hybrid_pchan05

else
 clear fn_int zprime
% ver=[ver];
% caseid=ver; caseid(1:5)=[];

% cd ../index_wise
 BlockStat_Hybrid_pchan03
end
%}

end  %m sksVerArray
%

% BlockStatT85NewReversalTM_pchan

%
% cd ../matlab
cd(oldpwd)
disp('finish read'); toc
%disp('pause');pause

%%
% caseid='ERA-interim_19790101-20171231'; season=true;
%ver=['0a13_',caseid];
%load(['../index_wise/wrk_',ver,'.mat']); clear text;
%disp('finish load wrk'); toc



%ver=['0a13_',caseid];
%fn_load2 = ['../matlab/label_int_z500_',ver,'.mat'];
%mat_load2 = matfile(fn_load2);
%load(fn_load2,'timeNan')
%PER0202 = (mat_load2.label>0);
%PER0202 = (mat_load2.label);

verH=[ver,'N'];
fn_load3 = ['../sks/label_int_z500_',verH,'.dat'];  % durthresh
fid3 = fopen(fn_load3);
PER0202 = single( fread(fid3,Inf,'single') );
fclose(fid3);
PER0202 = reshape(PER0202,128,64,[]);

%Wgt = mat_load2.zprime;

%{
%ver=['0702_',caseid];
ver=['07',ver(3:end)];
%fn_load2 = ['../matlab/label_int_z500_',ver,'.mat'];
%mat_load2 = matfile(fn_load2);
%load(fn_load2,'timeNan')
%PER0702 = (mat_load2.label>0);
%PER0702 = (mat_load2.label);

verH=[ver,'N'];
fn_load3 = ['../sks/label_int_z500_',verH,'.dat'];  % durthresh
fid3 = fopen(fn_load3);
PER0702 = single( fread(fid3,Inf,'single') );
fclose(fid3);
PER0702 = reshape(PER0702,128,64,[]);
%}
PER0702 = PER0202;sdJJA_SH=sdJJA_NH;  %TODO place holder

% fn_nco   = ['../sks/nco_z500_zg_day_',caseid,'.nc'];
%time = ncread(fn_nco,'time');

%ver=['0a13_',caseid];
%strTitle = 
addpath('/n/home05/pchan/bin');
load coastlines  % for plotting
%latt42=y; lont42=x;
lonplot = [lont42(:); 2*lont42(end)-lont42(end-1)];  % cyclic point added

fn_figure = ['../index_wise/all',ver,'.ps'];



%% quick polyfit, lagcorr
%
 %text='AOefbCV1HF'; thresh={0.01,'quantile',5,5,0}; caseid=[text,'T63h00'];
%thresh={0.01,'quantile',5,0,0}; caseid=['ERA-interim_19790101-20171231']; text=caseid;
%thresh={0.01,'quantile',5,5,0}; caseid=['ERA-interim_19790101-20171231'];
%caseid=['ERA-interim_19790101-20171231'];
%verX='x912';
%textTH=[num2str(thresh{1}), thresh{2}(1), num2str(thresh{4}),'K', num2str(thresh{3}),'d',num2str(thresh{5}),'lat'];
 %load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','ds','Cold_yxht','Hot_yxht','T850f_yxht','areaEarth')
 %load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh','yStart','yEnd','nyr' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend')
%load(['temp_',textTH,'_',text,'.mat'],'caseid','text','textTH','thresh' ,'ds','yStart','yEnd','nyr','fn_t42','latt42' ,'ds_jja','Hot_jja','mx2t_jja','hJJAstart','hJJAend','lsm_jja' ,'ds_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend','lsm_djf')
load(['temp_',verX,'_',caseid,'.mat'],'caseid','strTitleX','thresh' ,'ds','yStart','yEnd','nyr','f_h2d','fn_t42','latt42','areaEarth' ,'ds_jja','nd_jja','Hot_jja','mx2t_jja','HotQuantile','hJJAstart','hJJAend','lsm_jja','Hotid_jja','HotAttr' ,'ds_djf','nd_djf','Cold_djf','mn2t_djf','hDJFstart','hDJFend','lsm_djf','Coldid_djf','ColdAttr' ,'Hot_xyn')
%matX=matfile(['temp_',verX,'_',caseid,'.mat']); strTitleX=matX.strTitle;
%disp('finish load temp'); toc

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

%PER_jja = false(ds_jja);
PER_jja = zeros(ds_jja);
%Wgt_jja = zeros(ds_jja,'single');
tpointer = 1;
for yyyy = yStart:yEnd
  tstart = find(time==hJJAstart(yyyy-yStart+1));
  tend   = find(time==hJJAend(yyyy-yStart+1));
  PER_jja(:,:,tpointer+(0:tend-tstart)) = PER0202(:,:,tstart:tend);  % TODO
%  Wgt_jja(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end

%% collect DJF
%PER_djf = false(ds_djf);
PER_djf = zeros(ds_djf);
%Wgt_djf = zeros(ds_djf,'single');
tpointer = 1;
for yyyy = yStart+1:yEnd
  tstart = find(time==hDJFstart(yyyy-yStart));
  tend   = find(time==hDJFend(yyyy-yStart));
  PER_djf(:,:,tpointer+(0:tend-tstart)) = PER0702(:,:,tstart:tend);
%  Wgt_djf(:,:,tpointer+(0:tend-tstart)) = Wgt(:,:,tstart:tend);

  tpointer = tpointer +tend-tstart+1;
end
clear PER Wgt PER0202 PER0702

Wgt_jja(isnan(Wgt_jja))=0;
Wgt_djf(isnan(Wgt_djf))=0;
%disp('finish collect'); toc

PER_jja(:,latt42(:)<=thresh{5},:) = 0;
PERid_jja = PER_jja;  % TODO id from program
PER_jja = PER_jja>0;
PERid_djf = PER_djf;
PER_djf = PER_djf>0;

%%
mx2t_jja = mx2t_jja - repmat(max(HotQuantile,thresh{4}),[1 1 ds_jja(3)]);
Wgt_jja = Wgt_jja - prm.A* repmat([sdJJA_SH*ones(1,ds(2)/2), sdJJA_NH*ones(1,ds(2)/2)],[ds(1) 1 ds_jja(3)]);

Hot_n = [areaEarth * squeeze(mean(mean(reshape(Hot_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
Hotw_n = [areaEarth * squeeze(mean(mean(reshape(Hot_jja.*mx2t_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERwjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*Wgt_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERljja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*lsm_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
PERlwjja_n = [areaEarth * squeeze(mean(mean(reshape(PER_jja.*lsm_jja.*Wgt_jja,[ds(1:2),nd_jja,nyr]),3),1))]';
disp(sprintf('jja: D=%i A=%.1f S=%.2g R=%i %+.3f %+.3f %+.3f %+.3f',prm.D,prm.A,prm.S,prm.R, corr(PERjja_n(:),Hot_n(:)), corr(PERljja_n(:),Hot_n(:)) , corr(PERwjja_n(:),Hotw_n(:)), corr(PERlwjja_n(:),Hotw_n(:)) ));

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
%save(['../index_wise/scatter_',ver,'.mat'],'ver','strTitle','prm','timeNan','lont42','latt42', 'Hot_n','Hotw_n','PERjja_n','PERwjja_n','PERljja_n','PERlwjja_n', 'sdJJA_NH','sdJJA_SH','sdDJF_NH','sdDJF_SH','-v7.3');
toc

%{
%for prmd = 1:2:13
%  prm.D = prmd;
for pm = 1:8
  DArr = [1:2:7, 10:4:22];
  prm.D = DArr(pm);
  for prma = -0.5:0.5:4
    prm.A = prma;
    clearvars -except  prm rrr pm;
    BlockStat_d2d_pchan06;  % TODO
    rrr.r00( pm, (prm.A+1)*2 ) = corr(PERjja_n(:),Hot_n(:));
    rrr.r01( pm, (prm.A+1)*2 ) = corr(PERljja_n(:),Hot_n(:));
    rrr.r02( pm, (prm.A+1)*2 ) = corr(PERwjja_n(:),Hotw_n(:));
    rrr.r03( pm, (prm.A+1)*2 ) = corr(PERlwjja_n(:),Hotw_n(:));
  end
end
% fn_save  = ['corrTab_',verX,ver,'.mat'];
% save(fn_save,'prm','rrr','strTitle');
addpath('/n/home05/pchan/bin');
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(-0.5:0.5:4, 1:8,rrr.r01);
%colormap(gca,flipud(hot)); colorbar;
colormap(gca,b2r(-1,1)); colorbar;
  for pm = 1:8
    for prma = -0.5:0.5:4
      text(prma,pm,sprintf('%+.3f',rrr.r01(pm,(prma+1)*2)),'HorizontalAlignment','center','fontsize',16);
    end
  end
  title({strTitle},'FontSize',20); yticks(1:8); yticklabels([1:2:7, 10:4:22]);
%xticks(1:np); xticklabels(strTickLabel); xtickangle(45);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);
figure('units','inches','position',[0 1 11 8.5], 'paperUnits','inches','papersize',[11 8.5],'paperposition',[0 0 11 8.5]);
pcolorPH(-0.5:0.5:4, 1:8,rrr.r03);
%colormap(gca,flipud(hot)); colorbar;
colormap(gca,b2r(-1,1)); colorbar;
  for pm = 1:8
    for prma = -0.5:0.5:4
      text(prma,pm,sprintf('%+.3f',rrr.r03(pm,(prma+1)*2)),'HorizontalAlignment','center','fontsize',16);
    end
  end
  title({strTitle},'FontSize',20); yticks(1:8); yticklabels([1:2:7, 10:4:22]);
%xticks(1:np); xticklabels(strTickLabel); xtickangle(45);
xlabel('A'); ylabel('D');
axis ij; %axis square;
set(gca,'FontSize',14);
fn_figure = ['corrTable_',verX,'.ps'];
print(gcf, '-dpsc2','-append',fn_figure);
%}

%{
save(['wrk_0acr.mat'],'PER_jja','-v7.3');
load wrk_2a6x.mat
PER_jja(:,latt42(:)<=thresh{5},:) = false;
nnz(~PER_jja&PERcr)
%}

%
%% label contiguous, see blocklabel, m2d
if (exist('rrr','var')==0)
% method array
 %PERid_jja = repmat(reshape([1:ds(1)*floor(ds(2)/2)*nd_jja], [ds(1) floor(ds(2)/2) nd_jja]), [1 2 nyr]);
%PERid_jja = reshape([1:prod(ds_jja)], [ds_jja]);  %TODO
%PERid_jja = PERid_jja.*PER_jja;
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

%PERid_djf = reshape([1:prod(ds_djf)], [ds_djf]);
%PERid_djf = PERid_djf.*PER_djf;
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

save(['../index_wise/BlockFreq_',verX,ver,'.mat'],'ver','strTitle','prm','timeNan','lont42','latt42','PERid_jja','PERjjaAttr','PERid_djf','PERdjfAttr','PER_jja','PER_djf','Wgt_jja','Wgt_djf', 'Hot_n','Hotw_n','PERjja_n','PERwjja_n','PERljja_n','PERlwjja_n', 'sdJJA_NH','-v7.3');  %,'sdJJA_SH','sdDJF_NH','sdDJF_SH'
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

%print(gcf,'-dpdf',['../index_wise/scatter_',verX,ver,'.pdf']);
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

%print(gcf,'-dpdf',['../index_wise/scatter_',verX,ver,'.pdf']);
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
plotm(coastlat,coastlon,'k')
title({'JJA blocking frequency (%)'},'fontsize',16);
tightmap;

subplot(2,3,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERgfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitle,'matched blocking'},'fontsize',16);
tightmap;

subplot(2,3,3);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERbfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 8]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({'unmatched blocking'},'fontsize',16);
tightmap;

subplot(2,3,4);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*(Hotgfreq_jja+Hotbfreq_jja)'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({'JJA extreme frequency (%)'},'fontsize',16);
tightmap;

subplot(2,3,5);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*Hotgfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
plotm(coastlat,coastlon,'k')
title({strTitleX,'matched extreme'},'fontsize',16);
tightmap;

subplot(2,3,6);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[thresh{5}-2 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*Hotbfreq_jja'); shading flat;
colormap(gca,flipud(hot(10))); caxis([0 1.0]); colorbar; %caxis auto;  % TODO
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

%print(gcf,'-dpdf',['../index_wise/scatter_',verX,ver,'.pdf']);
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

%print(gcf,'-dpdf',['../index_wise/scatter_',verX,ver,'.pdf']);
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
  contour( Xedges(3:end),Yedges,Ncounts(3:end,:).^0.50','k');  % TODO
  contour( Xedges(1:2),Yedges,Ncounts(1:2,:).^0.50',3,'k');
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
  contour( Xedges(2:end),Yedges,Ncounts(2:end,:).^0.40','k');  % TODO
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
colormap(jet(10)); caxis([0 10]); colorbar; %caxis auto;  % TODO
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
colormap(jet(10)); caxis([0 10]); colorbar; %caxis auto;
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
colormap(jet(10)); caxis([0 10]*300); colorbar; %caxis auto;
plotm(coastlat,coastlon,'k');
title({strTitle,'JJA weighted blocking frequency (unit*%)'},'fontsize',16);
tightmap;

subplot(1,2,2);
axesm('MapProjection','eqaazim','origin',[90 0],'MapLatLimit',[0 90],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','stereo','origin',[90 0],'grid','on','mlinelocation',20,'plinelocation',20);
%axesm('MapProjection','pcarree','MapLatLimit',[0 90],'MapLonLimit',[-180 180]);
pcolormPC(latt42,lont42, 100*PERfreq_djf'); shading flat;
colormap(jet(10)); caxis([0 10]*300); colorbar; %caxis auto;
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
  colormap(gca,b2r(-2.5e-4,2.5e-4)); %colorbar; caxis auto;  %TODO
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
  colormap(gca,b2r(-2e-4,2e-4)); %colorbar; caxis auto;
  title({['DJF blocking regressed on extreme cold area'],'unit*%/K/km^2'},'fontsize',16);
else
  colormap(gca,b2r(-5e-6,5e-6)); %colorbar; caxis auto;
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
  colormap(gca,b2r(-50,30)); colorbar; %caxis auto;  %TODO
  title({strTitle,['JJA weighted blocking frequency trends'],'unit*%/yr'},'fontsize',16);
else
  colormap(gca,b2r(-0.25,0.15)); colorbar; %caxis auto;
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
  colormap(gca,b2r(-50,50)); colorbar; %caxis auto;  %TODO
  title({['DJF weighted blocking frequency trends'],'unit*%/yr'},'fontsize',16);
else
  colormap(gca,b2r(-0.2,0.15)); colorbar; %caxis auto;
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

