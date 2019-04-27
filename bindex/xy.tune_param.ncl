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
;  vernumArr = (/"5900","0a02","2937","4923","8900","5940","2946","4943"/)
;  titleArr = (/"P12","D13","H14","S06","M13","Modified P12","Modified H14","Modified S06"/)
  f1 = addfile("tune_x9190a13_ERA-interim_19790101-20171231.nc","r")
  e01        = f1->e01
  cm         = f1->cm
  mp         = f1->mp -1
;  np       = f1->np
  mq         = f1->mq -1
;  nq       = f1->nq
  ms         = f1->ms -1
;  ns       = f1->ns
;  titleArr   = str_right_strip(tostring(f1->titleArr))
  f2 = addfile("tune_x9180a13_ERA-interim_19790101-20171231.nc","r")
  e18        = f2->e01
  nr=1
;  e01&nq = nq
 ; list[0][0] expression dont work
  tablist = [/transpose(e01(:,:,mp(nr),nr)), e01(ms(nr),:,:,nr) /]
  tablist[0]@tiMainString  = "(a) "+f1@description +", when D="+tablist[0]@np+" days"
  tablist[1]@tiMainString  = "(b) "+f1@description +", when A="+tablist[1]@ns+" SD"
  tablist@wkPaperWidthF  = 6.
  tablist@wkPaperHeightF = 5.
  tablist@pdfname = get_script_prefix_name
  tablist@npan = (/1,2/)
  tablist@vpWidthF = 0.8
  tablist@lbLabelFontHeightF = 0.025

;  tablist2 = [/e01(ms(nr),:,:,1), e18(ms(nr),:,:,1), e01(ms(nr),:,:,2), e18(ms(nr),:,:,2), e01(ms(nr),:,:,3), e18(ms(nr),:,:,3)/]
  tablist2 = [/e01(ms(nr),:,:,1), e01(ms(nr),:,:,2), e01(ms(nr),:,:,3), e18(ms(nr),:,:,1), e18(ms(nr),:,:,2), e18(ms(nr),:,:,3)/]
;  do m = 0,ListCount(tablist2)-1
;    tablist2[m]@tiMainString  = "("+tostring(tochar(97+m))+") "
;  end do
  tablist2[0]@tiMainString  = "(a) "+str_right_strip(tostring(f1->regArr(1,:)))
  tablist2[1]@tiMainString  = "(b) "+str_right_strip(tostring(f1->regArr(2,:)))
  tablist2[2]@tiMainString  = "(c) "+str_right_strip(tostring(f1->regArr(3,:)))
  tablist2[3]@tiMainString  = "(d) "+str_right_strip(tostring(f2->regArr(1,:)))
  tablist2[4]@tiMainString  = "(e) "+str_right_strip(tostring(f2->regArr(2,:)))
  tablist2[5]@tiMainString  = "(f) "+str_right_strip(tostring(f2->regArr(3,:)))
  tablist2@wkPaperWidthF  = 8.5
  tablist2@wkPaperHeightF = 8.
  tablist2@pdfname = get_script_prefix_name+".sup"
  tablist2@npan = (/2,3/)
  tablist2@vpWidthF = 0.5
  tablist2@lbLabelFontHeightF = 0.015
 ; print(ListCount(tablist)+" "+dimsizes(tablist[1])+" "+ListCount(tablist))
  listlist = [/tablist, tablist2/]
;  listlist = [/ tablist2/]

  do ff = 0,ListCount(listlist)-1

  wrklist = listlist[ff]
  npan    = wrklist@npan
  type = "pdf"
; type@wkOrientation = "portrait" 
  type@wkPaperWidthF  = wrklist@wkPaperWidthF
  type@wkPaperHeightF = wrklist@wkPaperHeightF
  wks = gsn_open_wks(type,wrklist@pdfname)  ; TODO
;  gsn_define_colormap(wks,"WhBlGrYeRe")  ;BlWhRe
;  gsn_define_colormap(wks,cm)

; Set some basic resources
  res = True

  res@cnFillOn        = True
  res@cnFillMode      = "CellFill"
  res@cnFillColors    = cm
  res@cnCellFillEdgeColor = "Black"
  res@cnLinesOn       = False
  res@cnLineLabelsOn  = False
  res@cnInfoLabelOn   = False
 ;res@cnLineLabelFontHeightF = 0.01
 ;res@cnLevelSelectionMode = "EqualSpacedLevels"
 ;res@cnMaxLevelCount      = 99
  res@cnLevelSelectionMode = "ManualLevels"  ;"ExplicitLevels"  ;
  res@cnMinLevelValF       = 0.51
  res@cnMaxLevelValF       = 0.99
  res@cnLevelSpacingF      = 0.01
  res@lbLabelBarOn = False
;  res@lbBoxLinesOn         = False
;  res@lbBoxSeparatorLinesOn= False

;  res@sfXArray        = ispan(0,dimsizes(ns),1)-0.5  ; 1 more element
;  res@sfYArray        = ispan(0,dimsizes(nq),1)-0.5  ; 1 more element

  res@tmXBMode          = "Explicit"              ; explicit labels
;  res@tmXBValues        = ispan(0,dimsizes(ns)-1,1) ; location of labels 
;  res@tmXBLabels        = ns                    ; labels themselves
 ;res@tmXBLabelStride   = 24
; res@tmLabelAutoStride = True                    ; nice stride on labels
; res@tmXBFormat        = "0@!;*^se"
  res@tmXBLabelFontHeightF = 0.03
  res@tmXBLabelDeltaF      = -0.5
  res@tmYLMode          = "Explicit"              ; explicit labels
;  res@tmYLValues        = ispan(0,dimsizes(nq)-1,1) ; location of labels 
;  res@tmYLLabels        = nq                    ; labels themselves
  res@tmYLLabelFontHeightF = 0.03
  res@tmYLLabelDeltaF      = -0.5
; res@xyMonoDashPattern = True
 ;res@xyDashPatterns    = 1
; res@xyLineColors      = (/"Blue","Black"/)
;  res@xyMarkLineMode    = "Markers"
;  res@xyMarkers      = newindex
;  res@xyMarkerSizeF  = 0.015
 ;res@gsnXRefLine       = 0.5
 ;res@gsnXRefLineDashPattern = 1
;  res@tiXAxisString     = ns@description
;  res@tiYAxisString     = nq@description
  res@tiMainFontHeightF = 0.035
  res@tiXAxisFontHeightF= 0.035
  res@tiYAxisFontHeightF= 0.035
  res@tiDeltaF          = 1.
  res@gsnDraw         = False
  res@gsnFrame        = False
  res@gsnShape        = True
; res@gsnMaximize = True  ; panel
  res@gsnPaperOrientation  = "portrait"
 ;res@trYMaxF           = 1.0 ;0.8
 ;res@trYMinF           = 0

 ;res@vpHeightF       = 0.7
 ;res@vpWidthF        = 1

;  gsres@tfPolyDrawOrder  = "Draw"
  txres = True
  txres@txFontHeightF = 0.03

  plot := new(ListCount(wrklist), graphic)
;  dum2 = new(dimsizes(titleArr), graphic)
;  dum3 = new(dimsizes(titleArr), graphic)
;  figString = (/"a","b","c","d"/)

  do m = 0,ListCount(wrklist)-1
;    ver = vernumArr(m)+"_"+caseid
;    f1 = addfile("../index_wise/scatter_"+textTH+"_"+ver+".nc","r")

    tab := wrklist[m]
    dum  = new(dimsizes(tab), graphic)
    xArr:= tab&$tab!1$
    yArr:= tab&$tab!0$
    res@sfXArray        = ispan(0,dimsizes(xArr),1)-0.5  ; 1 more element
    res@sfYArray        = ispan(0,dimsizes(yArr),1)-0.5  ; 1 more element
    res@tmXBValues        = ispan(0,dimsizes(xArr)-1,1) ; location of labels 
    res@tmXBLabels        = xArr                    ; labels themselves
    res@tmYLValues        = ispan(0,dimsizes(yArr)-1,1) ; location of labels 
    res@tmYLLabels        = yArr                    ; labels themselves
    res@tiXAxisString     = xArr@description
    res@tiYAxisString     = yArr@description
    res@tiMainString  = tab@tiMainString

    res@tiXAxisOn            = (m.ge.npan(1)*(npan(0)-1))              ; panel 7
    res@tmXBLabelsOn         = (m.ge.npan(1)*(npan(0)-1))              ; panel 7
;    res@tmXBOn               = (m.ge.npan(1)*(npan(0)-1))              ; panel 7
    res@tiYAxisOn            = (mod(m,npan(1)).eq.0)
    res@tmYLLabelsOn         = (mod(m,npan(1)).eq.0)       ; panel 7
;    res@tmYLOn               = (mod(m,npan(1)).eq.0)       ; panel 7

;    plot(m) = gsn_csm_xy(wks,bjjaArr(m,:),ydata,res)
;    plot(m) = gsn_csm_xy(wks,transpose((/bjjaArr(m,:),bjjaArr(m,:)/))/1e6,transpose((/ydata,ydata/))/1e6,res)
 ; Each vector in the caXArray array must have at least 2 elements
    plot(m) = gsn_csm_contour(wks,tab,res)
    do nx = 0,dimsizes(xArr)-1
     do ny = 0,dimsizes(yArr)-1
       txres@txFontColor = "black"
       if (m.eq.0 .and. xArr(nx).eq.1.5 .and. yArr(ny).eq.2.5) then
         txres@txFontColor = "magenta"
       end if
       dum(ny,nx) = gsn_add_text(wks,plot(m),sprintf("%.2f",tab(ny,nx)),nx,ny,txres)
     end do
    end do

;    dum(m)  = gsn_add_polyline(wks,plot(m),(/0,max(bjjaArr(m,:))/)/1e6, ((/0,max(bjjaArr(m,:))/)*hotstat(0)+hotstat(1))/1e6, gsres)
;    txres = True
;    txres@txFontHeightF = 0.04
;    dum2(m)  = gsn_create_text(wks,figString(m),txres)
;    amres = True
;    amres@amZone = 0
;    amres@amJust = "TopLeft"
;    amres@amParallelPosF = -0.47
;    amres@amOrthogonalPosF = -0.47
;    dum3(m)  = gsn_add_annotation(plot(m),dum2(m),amres)

  end do
;draw(plot(0))
;frame(wks)
;exit
    resP = True
;   resP@txString    = "Time-step precip rate (mm/hr) at "+ times(it)
 ;  resP@txPosYF  = 0.98
    resP@gsnMaximize = True
    resP@gsnPaperOrientation   = "portrait"
;   resP@gsnPanelDebug = True
    resP@gsnPanelLabelBar = True
    resP@vpHeightF        = 0.10  ; labelbar
    resP@vpWidthF         = wrklist@vpWidthF  ; labelbar
;    resP@lbBoxLinesOn         = False
    resP@lbBoxSeparatorLinesOn= False
    resP@lbLabelFontHeightF    = wrklist@lbLabelFontHeightF
;    resP@lbLabelStride         = 5
;   resP@pmLabelBarOrthogonalPosF = -0.02   ; panel
;    resP@gsnPanelFigureStrings = "r="+sprintf("%.2f",escorc(bjjaArr,ydata))+" e~S~2~N~="+sprintf("%.2f",e01)
;    resP@amJust  = "BottomRight"
;    resP@gsnPanelFigureStringsFontHeightF = 0.02
    resP@gsnPanelScalePlotIndex   = 1    ; panel 24
    resP@gsnPanelXWhiteSpacePercent = 2  ; panel 6
    resP@gsnPanelYWhiteSpacePercent = 2  ; panel 6
 ;  resP@gsnPanelBottom = 0.14           ; panel 7
 ;  resP@gsnPanelTop    = 0.935          ; panel 7
    resP@gsnPanelLeft   = 0.13
 ;  resP@gsnFrame    = False
    gsn_panel(wks,plot,npan,resP)
 ;draw(plot)
 ;frame(wks)
  end do  ; ff

end
exit

; vim: set fdm=marker foldmarker=/;,;/:

