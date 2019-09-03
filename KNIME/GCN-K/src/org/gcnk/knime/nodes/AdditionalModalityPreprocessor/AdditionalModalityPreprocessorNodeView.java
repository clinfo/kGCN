package org.gcnk.knime.nodes.AdditionalModalityPreprocessor;

import org.knime.core.node.NodeView;

/**
 * <code>NodeView</code> for the "AdditionalModalityPreprocessor" Node.
 * Read new modality in CSV file format and generates output for AddModality node
 *
 * @author org.gcnk
 */
public class AdditionalModalityPreprocessorNodeView extends NodeView<AdditionalModalityPreprocessorNodeModel> {

    /**
     * Creates a new view.
     * 
     * @param nodeModel The model (class: {@link AdditionalModalityPreprocessorNodeModel})
     */
    protected AdditionalModalityPreprocessorNodeView(final AdditionalModalityPreprocessorNodeModel nodeModel) {
        super(nodeModel);

        // TODO instantiate the components of the view here.

    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void modelChanged() {

        // TODO retrieve the new model from your nodemodel and 
        // update the view.
        AdditionalModalityPreprocessorNodeModel nodeModel = 
            (AdditionalModalityPreprocessorNodeModel)getNodeModel();
        assert nodeModel != null;
        
        // be aware of a possibly not executed nodeModel! The data you retrieve
        // from your nodemodel could be null, emtpy, or invalid in any kind.
        
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void onClose() {
    
        // TODO things to do when closing the view
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void onOpen() {

        // TODO things to do when opening the view
    }

}

