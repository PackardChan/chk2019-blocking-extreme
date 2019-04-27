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
;  caseid="ERA-interim_19790101-20171231"
;  text=caseid
;  textTH="0.01q5K5d0lat_chunk"
;  vernumArr = (/"5900","0a02","2937","4923","8900","5940","2946","4943"/)
;  titleArr = (/"P12","D13","H14","S06","M13","Modified P12","Modified H14","Modified S06"/)
  f1 = addfile("scatter.nc","r")
  bjjaArr    = f1->bjjaArr
  ydata      = f1->ydata
  titleArr   = str_right_strip(tostring(f1->titleArr))
  e01        = f1->e01
;  e99        = f1->e99

  type = "pdf"
; type@wkOrientation = "portrait" 
  type@wkPaperWidthF  = 8.5
  type@wkPaperHeightF = 9.
  wks = gsn_open_wks(type,get_script_prefix_name)  ; TODO
  gsn_define_colormap(wks,"WhBlGrYeRe")  ;BlWhRe

; Set some basic resources
  res = True

; res@tmXBMode          = "Explicit"              ; explicit labels
; res@tmXBValues        = ispan(0,192,48)         ; location of labels 
; res@tmXBLabels        = lbl                     ; labels themselves
 ;res@tmXBLabelStride   = 24
; res@tmLabelAutoStride = True                    ; nice stride on labels
; res@tmXBFormat        = "0@!;*^se"
  res@tmXBLabelFontHeightF = 0.03
  res@tmYLLabelFontHeightF = 0.03
; res@xyMonoDashPattern = True
 ;res@xyDashPatterns    = 1
; res@xyLineColors      = (/"Blue","Black"/)
  res@xyMarkLineMode    = "Markers"
  newindex = NhlNewMarker(wks,sprinti("%02d",mod(ispan(1979,2017,1),100)), 0,0,0,1.3125,1,0)
  res@xyMarkers      = newindex
  res@xyMarkerSizeF  = 0.015
 ;res@gsnXRefLine       = 0.5
 ;res@gsnXRefLineDashPattern = 1
  res@tiXAxisString     = "Blocking area (10~S~6~N~ km~S~2~N~)"
  res@tiMainFontHeightF = 0.04
  res@tiXAxisFontHeightF= 0.04
  res@tiYAxisFontHeightF= 0.04
  res@gsnDraw         = False
  res@gsnFrame        = False
 ;res@gsnShape        = True
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

  gsres = True
  gsres@gsLineThicknessF = 8.
  gsres@gsLineColor      = "Red"
  gsres@tfPolyDrawOrder  = "Draw"

  plot = new(dimsizes(titleArr), graphic)
  dum  = new(dimsizes(titleArr), graphic)
  dum2 = new(dimsizes(titleArr), graphic)
  dum3 = new(dimsizes(titleArr), graphic)
  figString = (/"a","b","c","d"/)

  res@tiYAxisString     = "Hot extreme area (10~S~6~N~ km~S~2~N~)"
  do m = 0,dimsizes(titleArr)-1
;    ver = vernumArr(m)+"_"+caseid
;    f1 = addfile("../index_wise/scatter_"+textTH+"_"+ver+".nc","r")

;    bjjaArea_t = f1->bjjaArea_t
;    hotArea_t  = f1->hotArea_t
;    hotstat    = f1->hotstat

;  strXCorr = ['(correlation of trend = ',num2str(corr((yStart:yEnd)',bjjaArr(m,:)),'%+.2f'),')'];
;  strYCorr = ['(correlation of trend = ',num2str(corr((yStart:yEnd)',ydata(:)),'%+.2f'),')'];
    hotstat    = lspoly(bjjaArr(m,:),ydata,1.,2)
    hotstat    = hotstat(::-1)
    res@tiMainString  = titleArr(m)
;    if (escorc(bjjaArr(m,:),ydata).gt.0.2) then
;      gsres@gsLineColor      = "orange"
;    else
;      gsres@gsLineColor      = "green"
;    end if
;    if (m.le.3) then
;      res@tmBorderLineColor  = "red"
;    else if (m.eq.4) then
;      res@tmBorderLineColor  = "black"
;    else
;      res@tmBorderLineColor  = "cyan"
;    end if
;    end if

;    res@tmXBLabelsOn         = (m.ge.2)              ; panel 7
;    res@tmXBOn               = (m.ge.2)              ; panel 7
    res@tiYAxisOn            = (mod(m,2).eq.0)
    res@tmYLLabelsOn         = (mod(m,2).eq.0)       ; panel 7
;    res@tmYLOn               = (mod(m,2).eq.0)       ; panel 7

;    plot(m) = gsn_csm_xy(wks,bjjaArr(m,:),ydata,res)
    plot(m) = gsn_csm_xy(wks,transpose((/bjjaArr(m,:),bjjaArr(m,:)/))/1e6,transpose((/ydata,ydata/))/1e6,res)
 ; Each vector in the caXArray array must have at least 2 elements
    dum(m)  = gsn_add_polyline(wks,plot(m),(/0,max(bjjaArr(m,:))/)/1e6, ((/0,max(bjjaArr(m,:))/)*hotstat(0)+hotstat(1))/1e6, gsres)
    txres = True
    txres@txFontHeightF = 0.04
    dum2(m)  = gsn_create_text(wks,figString(m),txres)
    amres = True
    amres@amZone = 0
    amres@amJust = "TopLeft"
    amres@amParallelPosF = -0.47
    amres@amOrthogonalPosF = -0.47
    dum3(m)  = gsn_add_annotation(plot(m),dum2(m),amres)

  end do
    resP = True
;   resP@txString    = "Time-step precip rate (mm/hr) at "+ times(it)
 ;  resP@txPosYF  = 0.98
    resP@gsnMaximize = True
    resP@gsnPaperOrientation   = "portrait"
;   resP@gsnPanelDebug = True
;   resP@gsnPanelLabelBar = True
;   resP@lbLabelFontHeightF    = 0.012
    resP@gsnPanelFigureStrings = "r="+sprintf("%.2f",escorc(bjjaArr,ydata))+" e~S~2~N~="+sprintf("%.2f",e01)
    resP@amJust  = "BottomRight"
    resP@gsnPanelFigureStringsFontHeightF = 0.02
;   resP@pmLabelBarOrthogonalPosF = -0.02   ; panel
    resP@gsnPanelScalePlotIndex   = 1    ; panel 24
    resP@gsnPanelXWhiteSpacePercent = 5  ; panel 6
    resP@gsnPanelYWhiteSpacePercent = 5  ; panel 6
 ;  resP@gsnPanelBottom = 0.14           ; panel 7
 ;  resP@gsnPanelTop    = 0.935          ; panel 7
    resP@gsnPanelLeft   = 0.05
 ;  resP@gsnFrame    = False
    gsn_panel(wks,plot,(/2,2/),resP)
 ;draw(plot)
 ;frame(wks)

end
exit

; vim: set fdm=marker foldmarker=/;,;/:

