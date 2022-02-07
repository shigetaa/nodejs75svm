# サポートベクターマシンで文字認識をする
サポートベクターマシンとは、教師あり学習を用いるパターン認識モデルのひとつです。
識別や回帰分析へ適用できるものです。
ここでは、機械学習の利用例として、文字認識について学習してきます。

## 文字認識に挑戦
ここでは、Suport Vector Machine (SVM)を手書き文字の認識に利用してみたいと思います。
手書き文字の認識で最も簡単なのが数字の認識です。
大量の画像データを学習して、0から9までのいずれかに分類すると言うものです。

以下のサイトで、0から9までの手書き数字のデータ7000点が公開されており、機械学習やパターン認識に利用する事が出来ます。
各手書きの数字データは、縦横28x28ピクセルのデータとなってます。

[THE MNIST DATABASE of handwritten digits<br>http://yann.lecun.com/exdb/mnist/index.html](http://yann.lecun.com/exdb/mnist/index.html)

### 学習モデルを作る手順
SVMを利用して学習する手順は
1. パターンを学習させる
2. 学習データに基づいて手書き文字を判定させる

と言うものです。
そこで、手書き数字のデータベースから、パターン学習させるまでの手順は以下の様になります。

1. MNISTの手書き数字データをダウンロード
2. データを、扱いやすいように、CSVファイルに変換
3. CSVファイルを元に、学習データ(.svm)を作成する
4. 学習データをSVMに学習させ、モデル(.model)を作成する

### 手書き数字データをダウンロードする
MNISTのサイトで公開されているデータベースをダウンロードしましょう。
ダウンロードに必要なのは、以下の4つのファイルになります。

1. train-images-idx3-ubyte.gz
   学習用イメージ
2. train-labels-idx1-ubyte.gz
   学習用イメージのラベル
3. t10k-images-idx3-ubyte.gz
   テスト用イメージ
4. t10-labels-idx1-ubyte.gz
   テスト用イメージのラベル

ファイルは4つなのですが、実際は、学習用イメージ60,000点とテスト用イメージ10,000点の2つのファイルです。
これらは「.gz」形式で圧縮されているので、まずは解凍しますが、ここではプログラムで解凍するようにしたいので`kaitou.js`というファイルでプログラムを書いてみます。

Windows の場合は、コマンドで`gzip`が利用出来ないので、手動で解凍して頂戴。
```javascript
var fs = require('fs');
var exec = require('child_process').exec;

// ファイル一覧を得る
fs.readdir('.', function (err, files) {
	files.forEach(function (file) {
		// .gz以外は無視
		if (!/\.gz$/.test(file)) return;
		// 解凍
		console.log("file=" + file);
		var fn = file.replace(/\.gz$/, "");
		var cmd = [
			"gzip", "-dc",
			'"' + file + '"',
			'> "' + fn + '"'
		].join(" ");
		// 実行
		exec(cmd,
			function (err, stdout, stderr) {
				if (err) throw err;
				console.log(stdout, stderr);
			});
	});
});

```

プログラムを実行するには、以下のコマンドを入力します。

```bash
node kaitou.js
```

### バイナリーデーータをCSVに変換
ダウンロードしたデータを解凍したままでは、バイナリーデータなので、扱いやすいCSV形式に変換していきます。

バイナリーデータと言っても、それほど複雑な形式ではなく、先頭の16バイトがファイルヘッダーで、その後、28x28ピクセルデータが、60,000個続くというものになっています。

早速、バイナリーデータをCSVに変換するプログラム`mnistdb2csv.js`を作成していきます。

```javascript
// ファイル名を指定
var DIR_IMAGE = __dirname + "/image";

// モジュールの取り込み
var fs = require('fs');

// 変換処理
convertToCsv("train");
convertToCsv("t10k");

function convertToCsv(basename) {
	console.log("convert: " + basename);
	// 各種ファイル名を決定
	var file_images = basename + "-images-idx3-ubyte";
	var file_labels = basename + "-labels-idx1-ubyte";
	var file_csv = basename + ".csv";

	// ファイルを開く
	var f_img = fs.openSync(file_images, "r");
	var f_lbl = fs.openSync(file_labels, "r");
	var f_out = fs.openSync(file_csv, "w+");

	if (!fs.existsSync(DIR_IMAGE)) {
		fs.mkdir(DIR_IMAGE, function (err) {
			if (err) { throw err; }
			console.log("ディレクトリの作成が完了しました");
		});
	}

	// ヘッダを読む
	var buf_i = new Buffer(16);
	fs.readSync(f_img, buf_i, 0, buf_i.length);
	var buf_l = new Buffer(8);
	fs.readSync(f_lbl, buf_l, 0, buf_l.length);

	// ヘッダをチェック
	var magic = buf_i.readUInt32BE(0);
	var num_images = buf_i.readUInt32BE(4);
	var num_rows = buf_i.readUInt32BE(8);
	var num_cols = buf_i.readUInt32BE(12);
	var num_size = num_rows * num_cols;
	if (magic != 0x803) {
		console.error("[ERROR] Broken file=", magic.toString(16));
		process.exit();
	}
	console.log("num_of_images=" + num_images);
	console.log("num_of_rows=" + num_rows);
	console.log("num_of_cols=" + num_cols);
	console.log("num_of_pixel_size=" + num_size);

	// 画像を取り出す
	var buf_img = new Buffer(num_size);
	var buf_lbl = new Buffer(1);
	var mini_csv = "";
	for (var i = 0; i < num_images; i++) {
		// 経過を表示
		if (i % 1000 == 0) console.log(i + "/" + num_images);
		// 画像を読む
		var pos_i = i * num_size + 16;
		fs.readSync(f_img, buf_img, 0, num_size, pos_i);

		// ラベルを読む
		var pos_l = i * 1 + 8;
		fs.readSync(f_lbl, buf_lbl, 0, 1, pos_l);
		var no = buf_lbl[0];

		// PGM形式として保存 (テスト用)
		if (i < 30) {
			var s = "P2 28 28 255\n";
			for (var j = 0; j < 28 * 28; j++) {
				s += buf_img[j] + " ";
				s += (j % 28 == 27) ? "\n" : "";
			}
			var img_file =
				DIR_IMAGE + "/" + basename +
				"-" + i + "-" + no + ".pgm";
			fs.writeFileSync(img_file, s, "utf-8");
		}

		// CSVとして保存
		var cells = [];
		for (var j = 0; j < 28 * 28; j++) {
			cells.push(buf_img[j]);
		}
		s = no + "," + cells.join(",") + "\n";
		fs.writeSync(f_out, s, null, "utf-8");

		// テスト用のミニサイズCSVを作成
		if (i < 1000) {
			mini_csv += s;
			if (i == 999) {
				fs.writeFileSync(
					basename + "-mini.csv",
					mini_csv, "utf-8");
			}
		}
	}
	console.log("ok:" + basename);
}
```
プログラムを実行するには、以下のコマンドを入力します。

```bash
node mnistdb2csv.js
```
```bash
convert: train
num_of_images=60000
num_of_rows=28
num_of_cols=28
num_of_pixel_size=784
0/60000
...省略...
```
プログラムを実行すると、以下の4つのCSVファイルが出力されます。
1. train.csv
   手書きのデータ6万件をCSVに変換したもの
2. train-mini.csv
   上記から先頭の千件だけを抽出したもの
3. t10k.csv
   検証用の一万件の手書きデータをCSVに変換したもの
4. t10k-mini.csv
   上記から先頭の千件だけを抽出したもの

画像データの各ピクセルは、左上(0,0)から右下(27,27)へと 28x28 個順にカンマ区切りで並んだものとなってます。
