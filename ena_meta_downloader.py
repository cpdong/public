# inspired from ENAtools
import argparse,os,requests;

parser = argparse.ArgumentParser()
parser.add_argument('-id', metavar = 'accession', dest='acc_id', help='provide accession number or secondary accession number');

args = parser.parse_args();
acc_id = args.acc_id;

def download_ENA_meta(project_id, folder = ''):
    url = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=%s&result=read_run&fields=study_accession,secondary_study_accession,sample_accession,secondary_sample_accession,experiment_accession,run_accession,submission_accession,tax_id,scientific_name,instrument_platform,instrument_model,library_name,nominal_length,library_layout,library_strategy,library_source,library_selection,read_count,base_count,center_name,first_public,last_updated,experiment_title,study_title,study_alias,experiment_alias,run_alias,fastq_bytes,fastq_md5,fastq_ftp,fastq_aspera,fastq_galaxy,submitted_bytes,submitted_md5,submitted_ftp,submitted_aspera,submitted_galaxy,submitted_format,sra_bytes,sra_md5,sra_ftp,sra_aspera,sra_galaxy,cram_index_ftp,cram_index_aspera,cram_index_galaxy,sample_alias,broker_name,sample_title,nominal_sdev,first_created&format=tsv&download=true"%project_id
    r = requests.get(url, allow_redirects=True)
    filename = '%s_samples_info.tsv'%project_id
    filepath = (folder+'/'+filename).replace('//', '/') if folder != '' else filename
    if folder != '' and not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        f.write(r.content)

download_ENA_meta(acc_id)
