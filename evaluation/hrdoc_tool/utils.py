#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import glob
import re
from collections import Counter
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class2class = {
    "title": "title",
    "author": "author",
    "mail": "mail",
    "affili": "affili",
    "sec1": "section",
    "sec2": "section",
    "sec3": "section",
    "secx": "section",
    "alg": "paraline",
    "fstline": "fstline",
    "para": "paraline",
    "tab": "table",
    "fig": "figure",
    "tabcap": "caption",
    "figcap": "caption",
    "equ": "equation",
    "foot": "footer",
    "header": "header",
    "fnote": "footnote",
    "background": "table",
}

'''corsponding charactors are: Ĳ ĳ Œ œ ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ , e.g. ord(int(0x0132, 16)) = Ĳ '''
LIGATURES_EQUAL = {
    "0x0132": "IJ",
    "0x0133": "ij",
    "0x0152": "OE",
    "0x0153": "oe",
    "0xfb00": "ff",
    "0xfb01": "fi",
    "0xfb02": "fl",
    "0xfb03": "ffi",
    "0xfb04": "ffl",
    "0xfb05": "st"
}

REGULAR_EXPRESSIONS = {
    "documentclass":re.compile(r".*\\documentclass"),
    "begin_document":re.compile(r".*\\begin{document}"),
    "end_document":re.compile(r".*\\end{document}"),
}

LATEX_GREEK_MAP = {
    r'\alpha':'α', r'\Beta':'B', r'\delta':'δ', r'\Epsilon':'E', r'\Zeta':'Z', r'\theta':'θ', r'\iota':'ι', r'\Kappa':'K', r'\mu':'μ', 
    r'\Alpha':'A', r'\gamma':'γ', r'\Delta':'Δ', r'\varepsilon':'ε', r'\eta':'η', r'\Theta':'Θ', r'\Iota':'I', r'\lambda':'λ', r'\Mu':'M', 
    r'\beta':'β', r'\Gamma':'Γ', r'\epsilon':'ϵ', r'\zeta':'ζ', r'\Eta':'H', r'\vartheta':'ϑ', r'\kappa':'κ', r'\Lambda':'Λ', r'\nu':'ν', 
    r'\Nu':'N', r'\xi':'ξ', r'\Xi':'Ξ', r'\omicron':'ο', r'\Sigma':'Σ', r'\Tau':'T', r'\phi':'ϕ', r'\chi':'χ', r'\Chi':'X', r'\psi':'ψ',
    r'\Omicron':'O', r'\pi':'π', r'\Pi':'Π', r'\varpi':'ϖ', r'\varsigma':'ς', r'\upsilon':'υ', r'\Phi':'Φ', r'\Psi':'Ψ', r'	\Omega':'Ω',
    r'\rho':'ρ', r'\Rho':'P', r'\varrho':'ϱ', r'\sigma':'σ', r'\tau':'τ', r'\Upsilon':'Υ', r'\varphi':'φ', r'\omega':'ω', r'\\prime':'′',
}

REMOVE_LATEX_COMMAND = {
    'ref':r"\\ref{.*?}",
    'rm':r"\\rm ", 
}

BRACE_LATEX_COMMAND = r"\\\w*?\*?\{(.*?)\}"
TEXORPDFSTRING = r""
BRACE_LEFT_COMMAND = r"\\\w*?\*?\{"

class Left_Brace:
    """
        Used to logging one brace pair and their positions in text
    """
    def __init__(self, line_id, position, depth, right_brace_line = -1, right_brace_pos = -1):
        self.line_id = line_id
        self.position = position
        self.depth = depth # logging the nest depth of {}
        self.right_brace_line = right_brace_line
        self.right_brace_pos = right_brace_pos
    def set_right_brace(self, right_brace_line, right_brace_pos): # * logging crosponding closed } position
        assert right_brace_line >= self.line_id
        self.right_brace_line = right_brace_line
        self.right_brace_pos = right_brace_pos
    def get_braced_content(self, content_lines): # * getting the content closed by this Left_Brace node
        assert max(self.line_id, self.right_brace_line) < len(content_lines)
        assert min(self.right_brace_pos, self.position) > -1
        content = [] # * content is the text braced by the closed brace pair
        if self.line_id == self.right_brace_line:
            content.append(content_lines[self.line_id][self.position:self.right_brace_pos+1])
        else:
            content.append(content_lines[self.line_id][self.position:])
            content.extend(content_lines[self.line_id+1:self.right_brace_line])
            content.append(content_lines[self.right_brace_line][:self.right_brace_pos+1])
        return content
    def shift(self, shift):
        self.position += shift
        if self.right_brace_line == self.line_id:
            self.right_brace_pos += shift

class Begin_Brace:
    """
        Used to logging one begin-end brace pair and their positions in tex lines
    """
    def __init__(self, line_id, st, ed, depth, end_brace_line = -1, end_brace_st = -1, end_brace_ed = -1):
        self.line_id = line_id
        self.st = st
        self.ed = ed
        self.depth = depth # logging the nest depth of {}
        self.end_brace_line = end_brace_line
        self.end_brace_st = end_brace_st
        self.end_brace_ed = end_brace_ed
    def set_end_brace(self, end_brace_line, end_brace_st, end_brace_ed): # * logging crosponding closed } position
        assert end_brace_line >= self.line_id
        self.end_brace_line = end_brace_line
        self.end_brace_st = end_brace_st
        self.end_brace_ed = end_brace_ed
    def get_braced_content(self, content_lines): # * getting the content closed by this Left_Brace node
        assert max(self.line_id, self.end_brace_line) < len(content_lines)
        assert min(self.st, self.end_brace_st) > -1
        content = [] # * content is the text braced by the closed brace pair
        if self.line_id == self.end_brace_line:
            content.append(content_lines[self.line_id][self.st:self.end_brace_ed+1])
        else:
            content.append(content_lines[self.line_id][self.st:])
            content.extend(content_lines[self.line_id+1:self.end_brace_line])
            content.append(content_lines[self.end_brace_line][:self.end_brace_ed+1])
        return content

def merge_seperate_brace_blocks(all_continuous_text_blocks):
    all_text = "".join(all_continuous_text_blocks)
    all_blocks_length = [len(block) for block in all_continuous_text_blocks]
    assert sum(all_blocks_length) == len(all_text)
    

    def find_block(pos, all_blocks_length):
        assert pos < sum(all_blocks_length)
        cur_len = -1
        block_id = 0
        while cur_len < pos and block_id < len(all_blocks_length):
            cur_len += all_blocks_length[block_id]
            block_id += 1
        return block_id - 1
    def find_block_id(block_id, need_joined_blocks):
        exists_block_id = [block_id in x for x in need_joined_blocks]
        if sum(exists_block_id) == 0:
            return None
        elif sum(exists_block_id) == 1:
            return exists_block_id.index(True)
    
    all_left_brace_nodes = []
    left_brace_stack = []
    for c_i, c in enumerate(all_text):
        if c == '{':
            left_brace_stack.append(Left_Brace(c_i, len(left_brace_stack))) 
            # use len(left_brace_stack) to represent depth of {}, means how many out-nested {} are in the stack
        if c == '}':
            if len(left_brace_stack) == 0:
                logger.error("Text containing unclosed brace pair! {}".format(all_text))
                continue
            left_brace_stack[-1].set_right_brace(c_i)
            all_left_brace_nodes.append(left_brace_stack.pop())
    all_depth0nodes = [i for i in all_left_brace_nodes if i.depth==0]
    all_depth0nodes = sorted(all_depth0nodes, key=lambda x: x.position)
    need_joined_blocks = []
    for node in all_depth0nodes:
        start_block = find_block(node.position, all_blocks_length)
        end_block = find_block(node.right_brace_position, all_blocks_length)
        if start_block == end_block:
            continue
        else:
            need_joined_blocks.append(list(range(start_block, end_block+1)))
    if len(need_joined_blocks) == 0:
        return all_continuous_text_blocks
    need_joined_blocks = sorted(need_joined_blocks, key=lambda x: sum(x))
    right_i = len(need_joined_blocks)-1
    while right_i > 0:
        left_i = right_i - 1
        if len(set(need_joined_blocks[right_i]+need_joined_blocks[left_i])) != len(need_joined_blocks[right_i]) + len(need_joined_blocks[left_i]):
            need_joined_blocks[left_i] = list(set(need_joined_blocks[left_i] + need_joined_blocks[right_i]))
            need_joined_blocks.pop(right_i)
        right_i = left_i
    appended = [False]*len(need_joined_blocks)
    joined_text = [" ".join(all_continuous_text_blocks[min(x):max(x)+1]) for x in need_joined_blocks]
    joined_blocks = []
    for block_id, block in enumerate(all_continuous_text_blocks):
        find_block_id_ = find_block_id(block_id, need_joined_blocks)
        if find_block_id_ == None:
            joined_blocks.append(block)
        elif not appended[find_block_id_]:
            joined_blocks.append(joined_text[find_block_id_])
            appended[find_block_id_] = True
    # assert all_text.count('{') == all_text.count('}')
    return joined_blocks


def find_command_contents(tex_content, need_info_level="Nested_Text_Depth_0"):
    """Find all {content} saved in brace pairs

    Args:
        tex_content (string): plain string text from tex files
        need_info_level (string): how many infomation you want
            {
                "Nested_Text_Depth_0": Return only the texts in continuous brace pairs where depth=0,
                "Nested_Text_Depth_0_First": Return only the first text in continuous brace pairs where depth=0,
            }
    
    Exsampls:
        str = "[short name]{A fancy title}{A substitude title}"\n
        find_command_contents(str, "Nested_Text_Depth_0") -> ["A fancy title", "A substitude title"]\n
        find_command_contents(str, "Nested_Text_Depth_0_First") -> ["A fancy title"]
    """
    # * Test done!
    all_left_brace_nodes = []
    left_brace_stack = []
    for c_i, c in enumerate(tex_content):
        if c == '{':
            left_brace_stack.append(Left_Brace(c_i, len(left_brace_stack))) 
            # use len(left_brace_stack) to represent depth of {}, means how many out-nested {} are in the stack
        if c == '}':
            if len(left_brace_stack) == 0:
                logger.error("Text containing unclosed brace pair! {}".format(tex_content))
                continue
            left_brace_stack[-1].set_right_brace(c_i)
            all_left_brace_nodes.append(left_brace_stack.pop())
    while len(left_brace_stack) != 0:
        logger.warning("Text containing unclosed brace pair! Left brace pos at : {} in tex : {}".format(left_brace_stack[-1].position, tex_content))
        left_brace_stack[-1].set_right_brace(c_i) # c_i here should = len(tex_content)-1
        all_left_brace_nodes.append(left_brace_stack.pop())
    all_depth0nodes = [i for i in all_left_brace_nodes if i.depth==0]
    all_depth0nodes = sorted(all_depth0nodes, key=lambda x: x.position)
    assert len(all_depth0nodes) > 0, logger.error("No depth=0 brace node was found! tex : {}".format(tex_content))
    all_continuous_depth0nodes = [all_depth0nodes[0]]
    for next_node in all_depth0nodes[1:]:
        if all_continuous_depth0nodes[-1].is_continous_brace(next_node):
            all_continuous_depth0nodes.append(next_node)
        else:
            break
    if need_info_level == "Nested_Text_Depth_0":
        return [(i.get_braced_content(tex_content), i.position, i.right_brace_position) for i in all_continuous_depth0nodes]
    elif need_info_level == "Nested_Text_Depth_0_First":
        i = all_continuous_depth0nodes[0]
        return [(i.get_braced_content(tex_content), i.position, i.right_brace_position)]
    else:
        raise NotImplementedError

def remove_comment(code_str):
    """ Remove comment for each line in tex file """
    percent_start_idx = [each.start() for each in re.finditer('%', code_str)]
    percent_start_idx = list(filter(lambda x:True if x == 0 else code_str[x - 1] != '\\', percent_start_idx))
    no_comment_str = code_str
    if percent_start_idx:
        no_comment_str = code_str[:percent_start_idx[0]+1] # remove all string that appear after %
    no_comment_str = no_comment_str.strip()
    return no_comment_str

def open_tex_file(path):
    """ Read tex lines, transfer encoding when not using utf-8 """
    try:
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        lines = [remove_comment(l) for l in lines]
        f.close()
    except:
        logger.warning("This file containing no UTF-8 encoded text: {}".format(path))
        with open(path,'rb') as q:
            tmp = chardet.detect(q.read())
            encode_str = tmp['encoding']
        f = open(path, 'r', encoding=encode_str)
        lines = f.readlines()
        lines = [remove_comment(l) for l in lines]
        f.close()
    return lines

def count_latex_content(latex_path, content):
    lines = open_tex_file(latex_path)
    return " ".join([i.strip() for i in lines]).count(content)

def get_main_tex_path(dir):
    """ Find main tex file under latex folder `dir` """
    tex_paths = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file[-4:].lower() == '.tex':
                tex_paths.append(os.path.join(root, file))
    if not tex_paths:
        return ''
    main_tex_path_lst = []
    for path in tex_paths:
        lines = open_tex_file(path)
        documentclass_flag, begin_document_flag, end_document_flag = False, False, False
        for line in lines:
            documentclass_match = REGULAR_EXPRESSIONS['documentclass'].match(line)
            if documentclass_match != None:
                documentclass_flag = True
            begin_document_match = REGULAR_EXPRESSIONS['begin_document'].match(line)
            if begin_document_match != None:
                begin_document_flag = True
            end_document_match = REGULAR_EXPRESSIONS['end_document'].match(line)
            if end_document_match != None:
                end_document_flag = True
        if documentclass_flag and begin_document_flag and end_document_flag:
            main_tex_path_lst.append(path)

    # Should be main.tex if exists .bbl file with same name
    main_tex_path_lst_v2 = []
    if len(main_tex_path_lst) > 1:
        for tex_path in main_tex_path_lst:
            bbl_path = tex_path[:-4] + '.bbl'
            if os.path.exists(bbl_path):
                main_tex_path_lst_v2.append(tex_path)
        if len(main_tex_path_lst_v2) == 1:
            main_tex_path_lst = main_tex_path_lst_v2

    # Duplicated tex files with same name
    if len(main_tex_path_lst) == 2:
        if os.path.basename(main_tex_path_lst[0]) == os.path.basename(main_tex_path_lst[1]):
            if len(main_tex_path_lst[0]) < len(main_tex_path_lst[1]):
                main_tex_path_lst = [main_tex_path_lst[0]]
            else:
                main_tex_path_lst = [main_tex_path_lst[1]]

    if len(main_tex_path_lst) > 1:
        # Choose all.tex or main.tex as main tex file
        for path in main_tex_path_lst:
            if 'all.tex' in path.lower() or 'main.tex' in path.lower():
                return path
        # Sort tex files using the number using \section command
        main_tex_path_lst = sorted(main_tex_path_lst, key=lambda x: count_latex_content(x, r"\section"))
    
    if len(main_tex_path_lst) != 1:
        print('\ndir=', dir, '\nlen(main_tex_path_lst) = ', len(main_tex_path_lst), ', tex_path =', main_tex_path_lst)
    return main_tex_path_lst[0]

def translate_Unicode_latin_ligatures(unicode_chars):
    if len(unicode_chars) == 1 and hex(ord(unicode_chars)) in LIGATURES_EQUAL:
        return LIGATURES_EQUAL[hex(ord(unicode_chars))]
    else:
        return unicode_chars

def pdf2img(pdf_path, img_dir, zoom_x=1.0, zoom_y=1.0, rotate=0):
    import fitz
    print("Making dir: {}".format(img_dir))
    if not os.path.exists(img_dir):
        print("Making dir: {}".format(img_dir))
        os.makedirs(img_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdfDoc = fitz.open(pdf_path)
    img_paths = []
    for page_index in range(pdfDoc.pageCount):
        page = pdfDoc[page_index]
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)
        pix.writePNG(os.path.join(img_dir, '%s.png' % (page_index)))
        img_paths.append(os.path.join(img_dir, '%s.png' % (page_index)))
    return img_paths

def cal_wer(all_str, sub_str):
    if len(sub_str):
        return Levenshtein.distance(all_str, sub_str) / len(sub_str)
    else:
        return 1e6

def find_lcsubstr(s1, s2): 
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] 
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    return s1[p-mmax:p], mmax

def non_greed_wer(pred_str, label_str):
    origin_wer = cal_wer(pred_str, label_str)
    if len(pred_str) < 1/3 * len(label_str) or len(pred_str) > len(label_str) or\
        len(label_str) < 15 or len(pred_str) < 15:
        return origin_wer
    if origin_wer > 0.5:
        return origin_wer
    min_wer, best_l = origin_wer, len(pred_str)
    for l in range(len(pred_str) + 1, len(label_str)):
        temp_wer = cal_wer(pred_str, label_str[:l])
        if temp_wer < min_wer:
            min_wer = temp_wer
            best_l = l
    if min_wer < 0.3:
        return min_wer
    else:
        return origin_wer    

def visualize(pdf_path, img_dir, info_list, suffix):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    visual_img_dir = os.path.join(img_dir, 'visual')
    if not os.path.exists(visual_img_dir):
        os.mkdir(visual_img_dir)

    for page_index, info_pp in enumerate(info_list):
        img = cv2.imread(os.path.join(img_dir, '%s_%s.png' % (pdf_name, page_index)))
        for info in info_pp:
            cv2.rectangle(img, (info[1][0], info[1][1]), (info[1][2], info[1][3]), (0,0,255), 1)
        cv2.imwrite(os.path.join(visual_img_dir, '%s_%s_%s.png' % (pdf_name, page_index, suffix)), img)

def title_vis(img_path, coordinate_lst, texts=[], color=(255, 0, 0), suffix='_vis'):
    assert os.path.exists(img_path), '{} not exists'.format(img_path)
    coordinate_lst = np.array(coordinate_lst, dtype=np.int32)
    if coordinate_lst.size == 0:
        return
    coordinate_lst_poly = coordinate_lst[:, :, np.newaxis, :]
    img = cv2.imread(img_path)
    color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), 
                (0, 0, 0), (192, 192, 192), (0, 128, 255), (255, 0, 255)] 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    unit_len = 10
    num_box = coordinate_lst.shape[0]
    num_unit = int(num_box / unit_len) + 1
    for unit_i in range(num_unit):
        color_unit = color_lst[unit_i % len(color_lst)]
        cv2.polylines(img, pts=coordinate_lst_poly[unit_i*unit_len:(unit_i+1)*unit_len], isClosed=True, color=color_unit, thickness=2)  
    for text, box in zip(texts, coordinate_lst):
        cv2.putText(img, text, (box[2][0]+1, box[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    vis_img_path = os.path.join(os.path.split(img_path)[0], os.path.splitext(os.path.split(img_path)[1])[0] + suffix + os.path.splitext(os.path.split(img_path)[1])[1])
    cv2.imwrite(vis_img_path, img)

def pdf_text_log(lines_pdf, log_path):
    f = open(log_path, 'w', encoding='utf-8')
    for page_i, lines_page in enumerate(lines_pdf):
        f.write('\n\n***page {} ***\n'.format(page_i))
        for line in lines_page:
            f.write(line[0] + '\n', )
    f.close()    

def split_pdf_line_by_y(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        line_list = []
        
        for page in pdf.pages:
            line_list_pp = []
            line_x0, line_y0, line_x1, line_y1, line_contents, line_chars = \
                int(page.chars[0]['x0']), int(page.chars[0]['top']), int(page.chars[0]['x1']), int(page.chars[0]['bottom']), [], []
            for char in page.chars:
                # if char['text'] != '' and char['text'] != ' ' and len(char['text'].encode('gbk',errors='ignore').decode('gbk').encode('utf-8').decode('utf-8')) != 0:
                if char['text'] != '' and char['text'] != ' ' and len(char['text'].encode('utf-8').decode('utf-8')) != 0:
                    y = (int(char['top']) + int(char['bottom'])) / 2
                    if y < line_y1 and y > line_y0: # belong to the last line
                        line_x0, line_x1 = min(line_x0, int(char['x0'])), max(line_x1, int(char['x1']))
                        line_y0, line_y1 = min(line_y0, int(char['top'])), max(line_y1, int(char['bottom']))
                    else: # belong to a new line
                        if line_contents != [] and line_chars != []:
                            for idx in range(len(line_chars)):
                                line_chars[idx][-1][1] = line_y0
                                line_chars[idx][-1][3] = line_y1
                            line_list_pp.append([''.join(line_contents), [line_x0, line_y0, line_x1, line_y1], line_chars]) # process last line
                            line_x0, line_y0, line_x1, line_y1, line_contents, line_chars = \
                                int(char['x0']), int(char['top']), int(char['x1']), int(char['bottom']), [], [] # new line
                    chars = translate_Unicode_latin_ligatures(char['text'])
                    line_contents.append(chars)
                    line_chars.append([chars, char['fontname'], float(char['size']), [int(char['x0']), int(char['top']), int(char['x1']), int(char['bottom'])]])

            if line_contents != [] and line_chars != []:
                for idx in range(len(line_chars)):
                    line_chars[idx][-1][1] = line_y0
                    line_chars[idx][-1][3] = line_y1
                line_list_pp.append([''.join(line_contents), [line_x0, line_y0, line_x1, line_y1], line_chars])

            line_list.append(line_list_pp)
        return line_list

def split_pdf_line_by_x(line_list):
    block_list = []
    for page_index, lines_pp in enumerate(line_list):
        block_list_pp = []
        for line_pi in lines_pp:
            chars = line_pi[-1]
            pos = np.array([char[-1] for char in chars])
            delta = (pos[:, 2].mean() - pos[:, 0].mean() + pos[:, 3].mean() - pos[:, 1].mean()) / 2 * 5
            block = [chars[0]]
            for char in chars[1:]:
                x0, y0, x1, y1 = block[-1][-1]
                x0_, y0_, x1_, y1_ = char[-1]
                if abs(x0_ - x1) < delta:
                    block.append(char)
                else:
                    positions = np.array([c[-1] for c in block])
                    positions = [positions[:, 0].min(), positions[:, 1].min(), positions[:, 2].max(), positions[:, 3].max()]
                    block_list_pp.append([''.join([c[0] for c in block]), positions, block])
                    block = [char]

            positions = np.array([c[-1] for c in block])
            positions = [positions[:, 0].min(), positions[:, 1].min(), positions[:, 2].max(), positions[:, 3].max()]
            block_list_pp.append([''.join([c[0] for c in block]), positions, block])

        block_list.append(block_list_pp)
        
    block_list_s2 = []
    for page_index, lines_pp in enumerate(block_list):
        block_list_pp = []
        for l_i, line_pi in enumerate(lines_pp):
            chars = line_pi[-1]
            chars_font_type = [t[1] for t in chars]
            chars_font_counter = Counter(chars_font_type)
            char_font_most = chars_font_counter.most_common()
            char_font_most = [t[0] for t in char_font_most if t[1] >= 4]
            split_line = False
            if len(char_font_most) >= 2:
                extend_line_char_font_type = chars_font_type[0]
                if extend_line_char_font_type in char_font_most:
                    extend_line_char_font_type_bool = [t == extend_line_char_font_type for t in chars_font_type]
                    for extend_line_idx in range(1, len(chars_font_type)):
                        if not extend_line_char_font_type_bool[extend_line_idx]:
                            break
                    if chars[extend_line_idx - 1][0] == '.':
                        second_line_font_type = chars[extend_line_idx][1]
                        if second_line_font_type in char_font_most:
                            split_line = True
                            extend_line_end_idx = extend_line_idx - 1
                            if extend_line_end_idx < 8 and len(char_font_most) >= 2:
                                tid_chars_content = ''.join([c[0] for c in chars[:extend_line_end_idx + 1]]).strip()
                                match = re.compile('|'.join([r"(\d+\.)*", r"(XC|XL|L?X{0,3})(IX|XI*V?I*X?|IX|IV|V?I{0,3})[\.\:]?"]))
                                if match:
                                    len_tid_chars = extend_line_end_idx + 1
                                    second_extend_line_char_font_type = chars_font_type[len_tid_chars]
                                    chars_font_type = chars_font_type[len_tid_chars:]
                                    if second_extend_line_char_font_type in char_font_most:
                                        extend_line_char_font_type_bool = [t == second_extend_line_char_font_type for t in chars_font_type]
                                        for new_extend_line_idx in range(1, len(chars_font_type)):
                                            if not extend_line_char_font_type_bool[new_extend_line_idx]:
                                                break
                                        if chars[len_tid_chars + new_extend_line_idx - 1][0] == '.':
                                            third_line_font_type = chars[len_tid_chars + new_extend_line_idx][1]
                                            if third_line_font_type in char_font_most:
                                                extend_line_end_idx = extend_line_end_idx + new_extend_line_idx                                    
                                    
            
            if split_line:
                line_1_chars = chars[:extend_line_end_idx + 1]
                line_2_chars = chars[extend_line_end_idx + 1:]
                positions_1 = np.array([c[-1] for c in line_1_chars])
                positions_2 = np.array([c[-1] for c in line_2_chars])
                positions_1 = [positions_1[:, 0].min(), positions_1[:, 1].min(), positions_1[:, 2].max(), positions_1[:, 3].max()]
                positions_2 = [positions_2[:, 0].min(), positions_2[:, 1].min(), positions_2[:, 2].max(), positions_2[:, 3].max()]
                block_list_pp.append([''.join([c[0] for c in line_1_chars]), positions_1, line_1_chars])
                block_list_pp.append([''.join([c[0] for c in line_2_chars]), positions_2, line_2_chars])
            else:
                block_list_pp.append(line_pi)
        block_list_s2.append(block_list_pp)  
        
    for page_index, lines_pp in enumerate(block_list_s2):
        for line_pi in lines_pp:
            chars = line_pi[-1]
            pos = np.array([char[-1] for char in chars])
            left_char_x2 = pos[0][2]
            origin_content = line_pi[0]
            content_with_blank = ''
            black_idx_lst = []
            for char_i in range(1, len(chars)):
                char_str = chars[char_i][0]
                right_char_x1 = pos[char_i][0]
                if right_char_x1 - left_char_x2 > 0: # add blank
                    black_idx_lst.append(char_i)
                left_char_x2 = pos[char_i][2]
            for char_i, char in enumerate(chars):
                if char_i not in black_idx_lst:
                    content_with_blank = content_with_blank + char[0]
                else:
                    content_with_blank = content_with_blank + ' ' + char[0]
            line_pi[0] = content_with_blank
    

    return block_list_s2

def filter_lines(pdf_lines):
    new_block_list = []
    for page_lines in pdf_lines:
        filter_func = lambda x:len(x[0].replace(' ', '')) > 1 and\
            not (x[0].replace(' ', '').isdigit()) 
        new_page_lines = list(filter(filter_func, page_lines))
        new_block_list.append(new_page_lines)
    return new_block_list

def extract_pdf_line(pdf_path, visual=False):
    lines_pdf_y = split_pdf_line_by_y(pdf_path)
    lines_pdf_x = split_pdf_line_by_x(lines_pdf_y)
    lines_pdf_remove_tiny = lines_pdf_x 
    if visual:
        img_dir = pdf_path[:-4] + '_vis'
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        pdf2img(pdf_path, img_dir, zoom_x=1, zoom_y=1)
        img_paths = glob.glob(img_dir + '/*.png')
        img_paths = [p for p in img_paths if re.compile(r'.*?\d+\.png').match(p)]
        img_paths = sorted(img_paths, key=lambda x:int(re.compile(r'.*?(\d+)\.png').match(x).group(1)))
        # vis for split_pdf_line_by_y
        for img_i, img_path in enumerate(img_paths):
            lines_page_i = lines_pdf_y[img_i]
            box_lst = [[[line[1][0], line[1][1]], [line[1][2], line[1][1]], 
                        [line[1][2], line[1][3]], [line[1][0], line[1][3]]] for line in lines_page_i]
            title_vis(img_path, box_lst, suffix='_lines_y')
        # vis for split_pdf_line_by_x
        for img_i, img_path in enumerate(img_paths):
            lines_page_i = lines_pdf_x[img_i]
            box_lst = [[[line[1][0], line[1][1]], [line[1][2], line[1][1]], 
                        [line[1][2], line[1][3]], [line[1][0], line[1][3]]] for line in lines_page_i]
            title_vis(img_path, box_lst, suffix='_lines_x')
        # vis for remove_tiny
        for img_i, img_path in enumerate(img_paths):
            lines_page_i = lines_pdf_remove_tiny[img_i]
            box_lst = [[[line[1][0], line[1][1]], [line[1][2], line[1][1]], 
                        [line[1][2], line[1][3]], [line[1][0], line[1][3]]] for line in lines_page_i]
            title_vis(img_path, box_lst, suffix='_remove_tiny')
        pdf_text_log(lines_pdf=lines_pdf_remove_tiny, log_path=pdf_path[:-4] + '.txt')

    return lines_pdf_remove_tiny

def match_right_brace(input_str):
    assert input_str.startswith('{')
    num_left_brace, num_right_brace = 0, 0
    for c_i, c in enumerate(input_str):
        if c == '{':
            num_left_brace += 1
        if c == '}':
            num_right_brace += 1
        if num_left_brace == num_right_brace:
            break
    
    return input_str[:c_i + 1], c_i


def norm_latex_repr(latex_str):
    repr_latex = latex_str

    for k, v in REMOVE_LATEX_COMMAND.items():
        to_remove_latex_lst = re.findall(v, latex_str)
        for remove_str in to_remove_latex_lst:
            repr_latex = repr_latex.replace(remove_str, '')
    match_str = r'\texorpdfstring{'
    if match_str in repr_latex:
        start_idx_lst = [each.start() for each in re.finditer(r'\\texorpdfstring{', repr_latex)]
        texorpdf_brace_1_lst = []
        for s_i in start_idx_lst:
            texorpdf_brace_1, end_idx_1 = match_right_brace(repr_latex[s_i + len(match_str) - 1:])
            texorpdf_brace_1_lst.append(texorpdf_brace_1)
        for texorpdf_brace_1 in texorpdf_brace_1_lst:
            repr_latex = repr_latex.replace(texorpdf_brace_1, '')
        repr_latex = repr_latex.replace(r'\texorpdfstring', '')

    to_remove_latex_lst = re.findall(BRACE_LEFT_COMMAND, repr_latex)
    for to_remove in to_remove_latex_lst:
        repr_latex = repr_latex.replace(to_remove, '')

    repr_latex = repr_latex.replace('\\', '')

    repr_latex = repr_latex.replace('{', '').replace('}', '').replace('^', '').replace('_', '')

    return repr_latex



def repr_latex(latex_str):
    if '\\\\' in latex_str:
        pass
    latex_str = latex_str.replace('~', '')
    if len(latex_str.replace('{', '')) != len(latex_str.replace('}', '')):
        if not len(latex_str.replace('{', '')) + 1 == len(latex_str.replace('}', '')):
            pass
        else:
            latex_str = latex_str + '}'
    math_latex_lst = re.findall(r"\$.*?\$", latex_str)
    latex_repr = latex_str
    for math_latex in math_latex_lst:
        repr_math = math_latex
        for k, v in LATEX_GREEK_MAP.items():
            repr_math = repr_math.replace(k, v)
        repr_math = norm_latex_repr(repr_math)
        repr_math = repr_math.replace('$', '')
        latex_repr = latex_repr.replace(math_latex, repr_math)
    latex_repr = norm_latex_repr(latex_repr)   
    return latex_repr


def repr_pdf(pdf_content):
    pdf_content = pdf_content.strip()
    if pdf_content.lower().startswith('appendix'):
        pdf_content = pdf_content[len('appendix'):]
        pdf_content = pdf_content.strip()
    match_with_blank = r"[a-zA-Z][\.\:]? "
    match = re.match(match_with_blank, pdf_content)
    if match:
        group = match.group(0)
        pdf_content = pdf_content.replace(group, '')
    repr_pdf = pdf_content.replace(' ', '')
    remove_re_lst = '|'.join([r"(\d+[\.\:]?)+", r"(XC|XL|L?X{0,3})(IX|XI*V?I*X?|IX|IV|V?I{0,3})[\.\:]?"])
    match = re.match(remove_re_lst, repr_pdf)
    if match:
        group = match.group(0)
        repr_pdf = repr_pdf.replace(group, '')
    
    return repr_pdf

def tid_cal(tid):
    tid = tid.replace(' ', '')
    temp_tid = tid + '.0' * (3 - (len(tid) - len(tid.replace('.', ''))))
    all_n = temp_tid.split('.')
    rate = [1e3 ** i for i in range(len(all_n))][::-1]
    result = 0
    for n_i, n in enumerate(all_n):
        result += eval(n) * rate[n_i]

    return result


def trans_class(all_pg_lines, unit):
    if unit["class"] != "opara":
        return class2class[unit["class"]]
    else:
        parent_cl = all_pg_lines[unit['parent_id']]
        while parent_cl["class"] == 'opara':
            parent_cl = all_pg_lines[parent_cl['parent_id']]
        return class2class[parent_cl["class"]]