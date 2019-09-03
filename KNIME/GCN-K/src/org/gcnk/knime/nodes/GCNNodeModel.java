package org.gcnk.knime.nodes;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.def.StringCell;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeModel;
import org.knime.core.node.util.CheckUtils;

abstract public class GCNNodeModel extends NodeModel {
	
    protected GCNNodeModel(int nrInDataPorts, int nrOutDataPorts) {
        super(nrInDataPorts, nrOutDataPorts);
    }
	
    protected String getInPortFileNoCheck(final String fileKey, BufferedDataTable table) 
    {
    	String filename = "";
        int colIndex = table.getDataTableSpec().findColumnIndex(fileKey);
        for (DataRow row : table) {
        	if( !row.getKey().getString().equals("Files") )
        		continue;
            DataCell cell = row.getCell(colIndex);
            filename = ((StringCell)cell).getStringValue();
        }
        return filename;
    }

    protected String getInPortFile(final String fileKey, BufferedDataTable table) 
    		throws Exception {
    	String filename = getInPortFileNoCheck(fileKey, table);
        CheckFileExistence(filename, fileKey);
        return filename;
    }
    
    protected void CheckInportFile(final String fileKey, DataTableSpec tableSpec) throws InvalidSettingsException {
	    DataColumnSpec columnSpec = tableSpec.getColumnSpec(fileKey);
	    if (columnSpec == null ) {
	    	throw new InvalidSettingsException("Table contains no " + fileKey);
	    }
    }

    protected void CheckPythonPath() throws InvalidSettingsException {
	    String pythonPath = System.getenv("GCNK_PYTHON_PATH");
	    if( pythonPath == null || pythonPath.isEmpty() ) {
	        setWarningMessage("Set GCNK_PYTHON_PATH");
	    }
    }

    protected void CheckGCNKSourcePath() throws InvalidSettingsException {
	    String sourcePath = System.getenv("GCNK_SOURCE_PATH");
	    if( sourcePath == null || sourcePath.isEmpty() ) {
	        setWarningMessage("Set GCNK_SOURCE_PATH");
	    }
    }

    protected void CheckFileExistence(final String filename, final String fileKey)
    		throws InvalidSettingsException {
    	if( filename == null || filename.isEmpty() ) {
            setWarningMessage(fileKey + " not specified");
    	}
    	else {
	    	String warning = CheckUtils.checkSourceFile(filename);
	        if (warning != null) {
	            setWarningMessage(warning + fileKey);
	        }
    	}
    }

}
