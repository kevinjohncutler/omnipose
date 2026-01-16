from .imports import *

        
def load_links(filename):
    """
    Read a txt or csv file with label links. 
    These should look like::
        1,2 
        1,3
        4,7
        6,19
        ...
        
    Returns: 
        Links as a set of tuples. 
    """
    if filename is not None and os.path.exists(filename):
        links = set()
        with open(filename, "r") as file:
            lines = reader(file)
            for l in lines:
                # Check if the line is not empty before processing
                if l:
                    links.add(tuple(int(num) for num in l))
        return links
    else:
        return set()

def write_links(savedir,basename,links):
    """
    Write label link file. See load_links() for its output format. 
    
    Parameters
    ----------
    savedir: string
        directory in which to save
    basename: string
        file name base to which _links.txt is appended. 
    links: set
        set of label tuples {(x,y),(z,w),...}

    """
    with open(os.path.join(savedir,basename+'_links.txt'), "w",newline='') as out:
        csv_out = writer(out)
        for row in links:
            csv_out.writerow(row)
