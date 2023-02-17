# situation_detectection
昨年、幼稚園・保育園の幼稚園送迎バスでの死亡事故や閉じ込め事故が発生した。
それを防ぐためのシステムを開発できないかと思い実装を行った。

## プログラム実行手順及び、ファイル説明
前提条件として、「立っている」「座っている」「倒れている」のデータセットを作成してあること。
tf_pose_estimation_edit_fileは、tf-pose-estimationののtf_pose内で変更を加えたファイルのみを保存したものである。

1. git clone https://github.com/gsethi2409/tf-pose-estimation.git をgit cloneしローカルに保存
2. データセットの画像の骨骼検知を実施する。（tf_pose/run.pyを実行）
3. CNNのアルゴリズムを用いて、姿勢推定を行う。（cnn_folfer.py）
4. 学習したモデルを利用して、実際に姿勢推定を行う。（tf_pose/run_webcam.pyを実行）
※ make_pickle.pyとcnn_pickle.pyは、pickeleファイルを作成する際に使用する。

# 姿勢推定中のイメージ
run_webcam.pyを実行中の動作イメージ
![image](https://user-images.githubusercontent.com/67308009/219766547-70fefe2e-23dc-468d-a9a4-d4a445104e33.png)
