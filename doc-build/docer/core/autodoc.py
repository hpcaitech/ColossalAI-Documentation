import re
import doc_builder
import os

from typing import Dict

class AutoDoc:

    PATTERN = re.compile(r"{{ autodoc:(.+) }}")
    DEFAULT_IMPORT = "import {DocStringContainer, Signature, Divider, Title, Parameters, ExampleCode, ObjectDoc, Yields, Returns, Raises} from '@site/src/components/Docstring';"


    def convert_md_to_mdx(self, markdown_path: str, output_path: str, page_info: Dict[str, str] = None):
        """
        This method converts the auto-doc field in the markdown into React-based MDX file for docusaurus.
        """
        with open(markdown_path, 'r') as original_file:
            lines = original_file.readlines()
        
        # check if there if anything to convert
        contains_autodoc_field = False
        for line in lines:
            if re.match(self.PATTERN, line):
                contains_autodoc_field = True
                break

        # iterate over the lines to add the mdx code
        if contains_autodoc_field:
            new_lines = []
            
            # append the defualt import
            new_lines.append(self.DEFAULT_IMPORT)
            new_lines.append("\n\n")

            # iterate over the lines
            for i, line in enumerate(lines):
                if re.match(self.PATTERN, line):
                    # get the module name
                    obj_name = re.match(self.PATTERN, line).group(1)

                    # add the mdx code
                    pkg_name = obj_name.split('.')[0]
                    pkg = __import__(pkg_name)

                    docstring_mdx = doc_builder.autodoc(obj_name, pkg, page_info=page_info)
                    new_lines.append('\n')
                    new_lines.append(docstring_mdx)
                    new_lines.append('\n')
                else:
                    new_lines.append(line)
        
            with open(output_path, 'w') as output_file:
                output_file.writelines(new_lines)
    
    def convert_to_mdx_in_place(self, directory: str, page_info: Dict[str, str] = None):
        """
        This method converts the auto-doc field in the markdown into React-based MDX file for docusaurus.
        """
        for root, dirs, files in os.walk(directory):
            # apply auto-doc to files in-place
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    self.convert_md_to_mdx(file_path, file_path, page_info=page_info)
            
            for directory in dirs:
                self.convert_to_mdx_in_place(directory, page_info=page_info)
            







