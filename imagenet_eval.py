import os
import glob
import argparse
import requests


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepStreaks: eval on imagenet')

    parser.add_argument('--fetch', action='store_true', help='Fetch')
    parser.add_argument('--eval', action='store_true', help='Eval')

    args = parser.parse_args()

    if args.fetch:
        path_imagenet = './paper/imagenet'

        category_urls = glob.glob(os.path.join(path_imagenet, '*.txt'))

        for category_url in category_urls:
            category = os.path.basename(category_url).split('.txt')[0]
            print(category)

            path_category = os.path.join(path_imagenet, category)
            if not os.path.exists(path_category):
                os.mkdir(path_category)

            with open(category_url, 'r') as f:
                urls = f.read()
            urls = urls.split('\n')

            # print(urls[:2])

            ni = 1
            for url in urls:
                try:
                    if url.endswith('.jpg'):
                        r = requests.get(url)
                        if r.status_code == 200:
                            print(f'downloading image #{ni:04d} in {category} category')
                            with open(os.path.join(path_category, f'{ni:04d}.jpg'), 'wb') as f:
                                f.write(r.content)
                            ni += 1
                except Exception as e:
                    print(str(e))

    if args.eval:
        pass
