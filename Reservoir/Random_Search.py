#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大 田中勝規
#スペシャルサンクス：江波戸先輩
#作成日：2024/03/26
"""
本体

maru
"""

#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・save機能
#・MCの評価への対応
#・RSで出せる数値がfloat型のみな問題の解決
#・RS_Param配列の次元が統一されていないといけない問題の解決
#・並列処理

import numpy as np
import random

import Evaluation

class RandomSearch:
    def __init__(self, param, RSparam: dict):
        #パラメータ取得
        self.param = param
        self.search_ranges = RSparam                                                                            #RSするハイパーパラメータの範囲(種類 : 範囲: 2)
        
        self.best_error = np.inf                                                                                #最も良いerror保存用
        self.best_params = None                                                                                 #最も良いパラメータ保存用
        self.num_random_sample_stage = None                                                                     #RSを行うステージ数
        
        self.target_num_sample = self.param["RandomSearch_target_num_sample"]                                   #おおよその必要サンプル数
        self.RCmodel = self.param["Use_model"]                                                                  #用いるリザバーのモデル
        self.evaluation_indicator = self.param["Evaluation"]                                                    #評価指標
        self.evaluate_name = None                                                                               #評価指標の名前
        
    def calculate_num_random_sample_stage(self, target_num_sample: int):
        # サンプル数からステージ数を計算
        num: int = target_num_sample
        stage_num = 0
        while num > 0:
            stage_num += 1
            num -= 2 ** stage_num
        return stage_num

    def create_dict(self, keys_list, values_list):
        #差し替え用の辞書型生成
        result_dict = {}
        #'__'以前と以後で分ける
        for key, value in zip(keys_list, values_list):
            base_key, _ = key.split('__')
        
            if base_key not in result_dict:
                result_dict[base_key] = [value]
            else:
                result_dict[base_key].append(value)
    
        for key, value in result_dict.items():
            if len(value) == 1:
                result_dict[key] = value[0]
        
        return result_dict

    def generate_random_params(self, search_ranges: dict):
        #指定範囲でランダムなハイパーパラメータを生成する
        new_params_point = np.empty(len(search_ranges.values()))
        for i in range(len(list(search_ranges.values()))):
            new_params_point[i] = random.uniform(list(search_ranges.values())[i][0], list(search_ranges.values())[i][1])
        
        #差し替え用の新たな辞書型の作成
        random_dict = self.create_dict(list(search_ranges.keys()), new_params_point)
        
        return random_dict

    def replace_keys(self, new_param_dict, original_dict):
        #paramの中身を差し替え
        for key in original_dict.keys():
            if key in new_param_dict:
                original_dict[key] = new_param_dict[key]
        return original_dict

    
    def update_search_ranges_to_top_fifty_par_parameters(self, stage_sampled_params: np.ndarray, stage_sampled_errors: np.ndarray):
        # 探索範囲を上位50%に更新する
        sampled_errors = stage_sampled_errors.reshape(len(stage_sampled_errors),-1)
        merge_params = np.append(sampled_errors, stage_sampled_params, axis=1)
    
        sorted_params = merge_params[np.argsort(merge_params[:, 0])]
        
        sorted_params = sorted_params[:int(len(sorted_params)*0.5)]
    
        new_search_range_list = np.empty((sorted_params.shape[1], 2))
        for i in range(0, sorted_params.shape[1]):
            new_search_range_list[i] = [np.amax(sorted_params[:, i]), np.amin(sorted_params[:, i])]
    
        new_search_range_dict = {}
    
        for key, value in zip(list(self.search_ranges.keys()), new_search_range_list):
            new_search_range_dict[key] = value
    
        return new_search_range_dict
    
    def random_search(self):
        # ランダムサーチの実行
        # 実行回数からステージ数を決定
        self.num_random_sample_stage = self.calculate_num_random_sample_stage(self.target_num_sample)

        #初期化 initでしてるから不要かも
        self.best_error = np.inf
        self.best_params = None

        # ステージ数からサンプル数を計算
        num_all_samples = sum([2**i for i in range(self.num_random_sample_stage, 0, -1)])

        # サンプルが1e6(100万を超える場合警告)
        if num_all_samples > 1e6:
            raise ValueError('The number of samples is too large. This may take a long time to execute.')

        # ランダムサーチ開始
        progress_step = 1
        
        for stage in range(1, self.num_random_sample_stage + 1):
            num_samples = 2**(self.num_random_sample_stage + 1 - stage)

            # サンプル数 * 探索するパラメータの種類のempty(組み合わせ保存用)
            stage_sampled_params = np.empty((num_samples, len(list(self.search_ranges.keys()))))
            # サンプル数のempty(error保存用)
            stage_sampled_errors = np.empty(num_samples)
            
            # ステージ内でnum_samples回実行
            for i in range(num_samples):
                
                # 進捗表示
                print("\r Random search progress: %d / %d"%(progress_step, num_all_samples), end = "")
                
                np.random.seed()

                # 指定範囲からハイパーパラメータを生成
                new_params = self.generate_random_params(self.search_ranges)
                self.param = self.replace_keys(new_params, self.param)
                
                #評価指標に合わせて計算
                if self.evaluation_indicator == "NRMSE":
                    self.evaluate_name = "NRMSE"
                    # パラメータをモデルに与える(接続部)
                    return_params = Evaluation.Evaluation_NRMSE(self.param)()

                    # 評価(NRMSE等を返す)(接続部)
                    current_error = return_params["NRMSE_R_NRMSE"]

                    # bestを更新
                    if current_error < self.best_error:
                        self.best_error = current_error
                        self.best_params = new_params

                # パラメータと値を保存しておく
                stage_sampled_params[i] = np.array(list(new_params.values())).reshape(-1)
                stage_sampled_errors[i] = current_error

                # 進行度の更新
                progress_step += 1
                
            # 範囲を上位50%に更新
            """変更中：不等号の向きを逆にした。（実行中、徐々に精度悪化するため。"""
            if stage > self.num_random_sample_stage:
                self.search_ranges = self.update_search_ranges_to_top_fifty_par_parameters(stage_sampled_params, stage_sampled_errors)

        print('\n')
        print(self.evaluate_name + " : ", end="")
        print(self.best_error)
        print("best_param : ", end="")
        print(self.best_params)
