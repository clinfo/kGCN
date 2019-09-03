package org.gcnk.knime.nodes.AddModality;

import org.knime.core.node.NodeView;

/**
 * <code>NodeView</code> for the "AddModality" Node.
 *  * nAdd the new modality data to the GCN Dataset with the output of the GCNDatasetBuilder node and the output from the AdditionalModalityPreprocessor node as input
 *
 * @author org.gcnk
 */
public class AddModalityNodeView extends NodeView<AddModalityNodeModel> {

    /**
     * Creates a new view.
     * 
     * @param nodeModel The model (class: {@link AddModalityNodeModel})
     */
    protected AddModalityNodeView(final AddModalityNodeModel nodeModel) {
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
        AddModalityNodeModel nodeModel = 
            (AddModalityNodeModel)getNodeModel();
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

