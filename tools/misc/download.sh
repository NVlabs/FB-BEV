# Download zip dataset from Google Drive
filename='dd3d_det_final.pth'
# https://drive.google.com/file/d/158ltbC_wjRoe3uBnktbwCgeIByadwxTY/view?usp=share_link
# https://drive.google.com/file/d/1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A/view?usp=share_link
fileid='1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A'
wget --load-cookies /tmp.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename}
