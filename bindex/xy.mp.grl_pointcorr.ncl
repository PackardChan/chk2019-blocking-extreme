; polar map of blocking climatologies across models
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

; PROBLEMS
; maths: 
; mapping: 
; color: 

; :'<,'>s/[\[\];]//gce | '<,'>s/'/"/gce | '<,'>s?%?;?gce | '<,'>s/,/+/gce | '<,'>s/single/float/gce | '<,'>s/length/dimsizes/gce | '<,'>s/for/do/gce | '<,'>s/:/,/gce | noh

; :s/= /&addfile(/ce | s/^/,"r")/ce | noh

; :s/ncread(\(.*\),'\(.*\)')/\1->\2/c

; :s?zeros(\(.*\),'\(.*\)')?new((/\1/),\2)?

begin
;  caseid="ERA-interim_19790101-20151231"
  mArr := (/3,2/)
  f1 = addfile("pointcorr.nc","r")
  titleArr   = str_right_strip(tostring(f1->titleArr))
  corr_xym   = f1->corr_xym
  corr_xym@_FillValue = default_fillvalue("float")
  replace_ieeenan(corr_xym,corr_xym@_FillValue,0)
  rCorr_m    = f1->rCorr_m
  rCorr2_m   = f1->rCorr2_m

  type = "pdf"
; type@wkOrientation = "portrait" 
  type@wkPaperWidthF  = 16.
  type@wkPaperHeightF = 7.5
  wks = gsn_open_wks(type,get_script_prefix_name)  ; TODO
  gsn_define_colormap(wks,"MPL_rainbow")  ;BlWhRe

; Set some basic resources
  res = True

; res@xyMonoDashPattern = True
 ;res@xyDashPatterns    = 1
; res@xyLineColors      = (/"Blue","Black"/)
  res@xyMarkLineMode    = "Markers"
  newindex = NhlNewMarker(wks,str_sub_str(titleArr,"tuned ","t"), 0,0,0,1.3125,1,0)
  res@xyMarkers      = newindex
  res@xyMarkerSizeF  = 0.015
 ;res@gsnXRefLine       = 0.5
 ;res@gsnXRefLineDashPattern = 1
  res@tiXAxisString     = "Weighted mean local correlation"  ;"Mean local correlation weighted by area*std(block)"
  res@tiYAxisString     = "Hemispheric correlation"
;  res@tiMainFontHeightF = 0.04
  res@tiXAxisFontHeightF= 0.04
  res@tiYAxisFontHeightF= 0.04
  res@tmXBLabelFontHeightF = 0.035
  res@tmYLLabelFontHeightF = 0.035
  res@gsnDraw         = False
  res@gsnFrame        = False
  res@gsnShape        = True
; res@gsnMaximize = True  ; panel
  res@gsnPaperOrientation  = "portrait"
 ;res@trYMaxF           = 1.0 ;0.8
 ;res@trYMinF           = 0

 ;res@vpHeightF       = 0.7
 ;res@vpWidthF        = 1
/;
 ;res@pmLegendDisplayMode    = "Always"            ; turn on legend
 ;res@pmLegendZone           = 1
;;res@pmLegendSide           = "Right"             ; Change location of 
 ;res@pmLegendParallelPosF   = 0.98                ; move
;;res@pmLegendOrthogonalPosF = -0.2                ; move
 ;res@lgJustification        = "BottomRight"
 ;res@pmLegendWidthF         = 0.12                ; Change width and
 ;res@pmLegendHeightF        = 0.1                 ; height of legend.
;;res@lgPerimOn              = False               ; turn off box around
;;res@lgLabelFontHeightF     = .02                 ; label font height
 ;res@xyExplicitLegendLabels = (/"0.1a","a"/)         ; create explicit labels
;/

  plot = new(3, graphic)

  do m = 0,dimsizes(mArr)-1

    mpres = True
    mpres@cnFillOn        = True
    mpres@cnFillMode      = "CellFill"
    mpres@cnLinesOn       = False
    mpres@cnLineLabelsOn  = False
    mpres@cnLevelSelectionMode = "ManualLevels"  ;"ExplicitLevels"  ;
    mpres@cnMinLevelValF       = -0.9
    mpres@cnMaxLevelValF       = 0.9
    mpres@cnLevelSpacingF      = 0.1

;    mpres@tiMainString  = titleArr(mArr(m))
;    mpres@tiMainFontHeightF = 0.035

    mpres@gsnPolar = "NH"
    mpres@mpProjection = "LambertEqualArea"
    mpres@gsnPolarLabelSpacing = 60
    mpres@gsnDraw         = False
    mpres@gsnFrame        = False

    mpres@mpLimitMode        = "LatLon"
    mpres@mpMinLatF          = 35.
    mpres@lbLabelBarOn = False

    plot(m) = gsn_csm_contour_map_polar(wks,corr_xym(mArr(m),:,:),mpres)
  end do
    resP = True
;    resP@txString    = "Point-wise interannual Pearson correlation with hot extreme"
 ;  resP@txPosYF  = 0.98
    resP@gsnMaximize = True
    resP@gsnPaperOrientation   = "portrait"
;    resP@gsnPanelDebug = True
    resP@gsnPanelLabelBar = True
    resP@vpWidthF         = 0.5  ; labelbar
;   resP@lbLabelFontHeightF    = 0.012
    resP@lbTitleString         = "Point-wise correlation with hot extreme"
    resP@lbTitleFontHeightF    = 0.018
    resP@gsnPanelFigureStrings = (/"(a) ","(b) "/)+titleArr(mArr)
    resP@amJust  = "TopLeft"
    resP@gsnPanelFigureStringsFontHeightF = 0.018
    resP@pmLabelBarOrthogonalPosF = -0.01   ; panel
    resP@gsnPanelScalePlotIndex   = 1    ; panel 24
;   resP@gsnPanelYWhiteSpacePercent = 5  ; panel 6
 ;  resP@gsnPanelBottom = 0.14           ; panel 7
    resP@gsnPanelRight  = 0.65            ; panel 7
    resP@gsnFrame    = False
  gsn_panel(wks,plot(0:1),(/1,2/),resP)

  plot(2) = gsn_csm_xy(wks,transpose((/rCorr2_m(:dimsizes(rCorr_m)-2),rCorr2_m(:dimsizes(rCorr_m)-2)/)),transpose((/rCorr_m(:dimsizes(rCorr_m)-2),rCorr_m(:dimsizes(rCorr_m)-2)/)),res)
    resP2 = True
;    resP2@txString    = "Point-wise interannual Pearson correlation with hot extreme"
 ;  resP2@txPosYF  = 0.98
    resP2@gsnMaximize = True
    resP2@gsnPaperOrientation   = "portrait"
;    resP2@gsnPanelDebug = True
;    resP2@gsnPanelLabelBar = True
    resP2@gsnPanelFigureStrings = "(c)"
    resP2@amJust  = "TopLeft"
    resP2@gsnPanelFigureStringsFontHeightF = 0.018
;   resP2@pmLabelBarOrthogonalPosF = -0.02   ; panel
;   resP2@gsnPanelScalePlotIndex   = 1    ; panel 24
;   resP2@gsnPanelYWhiteSpacePercent = 5  ; panel 6
 ;  resP2@gsnPanelBottom = 0.14           ; panel 7
    resP2@gsnPanelLeft   = 0.67            ; panel 7
    resP2@gsnFrame    = False
  gsn_panel(wks,plot(2),(/1,1/),resP2)

  frame(wks)
end
exit

; vim: set fdm=marker foldmarker=/;,;/:

