# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import os


def create_html(tgt_file, num_items, caption_lst):
    body_text = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <link rel="stylesheet" href="/css/style.css">

        <title>CSS Grids Gallery</title>
    </head>
    <body>
        <div class="container">
            <div class="gallery">
"""
    for item_num in range(num_items):
        if item_num == len(caption_lst):
            break
        caption = caption_lst[item_num][1]
        fname = os.path.basename(caption_lst[item_num][0])
        body_text += f"""
            <figure class="gallery__item">
               <img src="rank0/{fname}" onclick="showText('{caption}')" class="gallery__img">
            </figure>
        """

    body_text += f"""</div></div>
                 <div id="caption" class="center">Caption</div>
                 </body></html>
                 <script language="JavaScript">
                     function showText(text){{
                        document.getElementById("caption").innerText = text;
                     }}
                 </script>
                 """
    f = open(tgt_file, "w")
    f.write(body_text)
    f.close()


def create_css(fname):
    contents = """
*,
*::after,
*::before {
  margin: 0;
  padding: 0;
  box-sizing: inherit;
}

.center {
  border: 5px solid;
  margin: auto;
  width: 100%;
  padding: 0px;
  font-size: large;
  text-align: center
}

html {
  box-sizing: border-box;
  font-size: 62.5%;
}

body {
  font-family: "Nunito", sans-serif;
  color: #333;
  font-weight: 300;
  line-height: 1.6;
}

.container {
  width: 100%;
  margin: 0.1rem auto;
}

.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(64px, 1fr));
  grid-auto-rows: 64px;
  gap: 0.2rem;
}

.gallery__img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
"""
    f = open(fname, "w")
    f.write(contents)
    f.close()
