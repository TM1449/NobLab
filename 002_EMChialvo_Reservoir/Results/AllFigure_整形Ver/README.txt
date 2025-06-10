スケールサイズは全て任意

#ルンゲクッタ法通常ローレンツ方程式 1/20 → 1/50
            "Task_NormalLorenz_Scale" : 1 / 50,
            "Task_NormalLorenz_Dt" : 0.005,
            "Task_NormalLorenz_Tau" : 5,
            "Task_NormalLorenz_InitTerm" : 1000,

            "Task_NormalLorenz_Sigma" : 10,
            "Task_NormalLorenz_Beta" : 8/3,
            "Task_NormalLorenz_Rho" : 28,



#ルンゲクッタ法通常レスラー方程式 1/20 → 1/10
            "Task_NormalRosslor_Scale" : 1 / 20,
            "Task_NormalRosslor_Dt" : 0.02,
            "Task_NormalRosslor_Tau" : 5,
            "Task_NormalRosslor_InitTerm" : 1000,

            "Task_NormalRosslor_a" : 0.2,
            "Task_NormalRosslor_b" : 0.2,
            "Task_NormalRosslor_c" : 5.7,


#ローレンツ方程式96のパラメータ 1/20 → 1/10
        "Task_Lorenz96_Scale" : 1 / 20,                       #ローレンツ方程式96の大きさ
        "Task_Lorenz96_Dt" : 0.005,                          #時間刻み幅
        "Task_Lorenz96_Tau" : 5,                            #どれくらい先を予測するか
        "Task_Lorenz96_InitTerm" : 1000,                    #初期状態排除期間
        "Task_Lorenz96_N" : 10,                             #ニューロン数
        "Task_Lorenz96_F" : 8,                              #大きさ？


#連続時間のマッキー・グラス方程式のパラメータ
            "Task_PredictDDE_Tau" : 5,                          #どれくらい先を予測するか
            "Task_MackeyGlassDDE_Dt" : 0.2,                    #時間刻み幅
            
            "Task_MackeyGlassDDE_Beta" : 0.2,                       #β:0.2
            "Task_MackeyGlassDDE_Gamma" : 0.1,                     #γ:0.1
            "Task_MackeyGlassDDE_N" : 10,                       #乗数: 10
            "Task_MackeyGlassDDE_Tau" : 22,                      #マッキー・グラスの遅延量 :17
            "Task_MackeyGlassDDE_InitTerm" : 1000,              #初期状態排除期間